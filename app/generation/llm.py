"""LLM provider abstraction — supports Gemini and OpenAI with clean swapping."""

from __future__ import annotations

import json
import httpx

from app.config import get_settings
from app.generation.prompts import build_rag_prompt
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("generation.llm")


class LLMProvider:
    """Abstract LLM provider interface with Gemini and OpenAI backends."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(
        self,
        query: str,
        context_passages: list[dict],
    ) -> str:
        """Generate a grounded answer from query + retrieved context.

        Args:
            query: User question
            context_passages: List of dicts with 'text', 'score', 'chunk_id'

        Returns:
            Generated answer string
        """
        prompt = build_rag_prompt(query, context_passages)

        if self.settings.llm_provider == "gemini":
            return await self._generate_gemini(prompt)
        elif self.settings.llm_provider == "openai":
            return await self._generate_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    async def _generate_gemini(self, prompt: str) -> str:
        """Call Google Gemini API with automatic fallback models."""
        api_key = self.settings.gemini_api_key or self.settings.llm_api_key
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or LLM_API_KEY not configured")

        primary_model = self.settings.gemini_model or "gemini-3.5-flash"
        fallback_models = [primary_model, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
        # Deduplicate while preserving order
        candidate_models = list(dict.fromkeys(fallback_models))

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.settings.llm_temperature,
                "maxOutputTokens": self.settings.llm_max_tokens,
            },
        }

        last_error = None
        with Timer("Gemini LLM generation") as t:
            for model in candidate_models:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={api_key}"
                )
                try:
                    async with httpx.AsyncClient(
                        timeout=self.settings.llm_timeout
                    ) as client:
                        response = await client.post(
                            url,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                        )
                except httpx.TimeoutException:
                    last_error = ConnectionError("Gemini API request timed out")
                    continue
                except httpx.ConnectError:
                    last_error = ConnectionError("Could not connect to Gemini API")
                    continue

                if response.status_code == 200:
                    try:
                        data = response.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        log.info(f"Gemini ({model}) generated {len(text)} chars ({t.elapsed_ms:.0f}ms)")
                        return text.strip()
                    except (KeyError, IndexError, json.JSONDecodeError) as e:
                        last_error = RuntimeError(f"Failed to parse Gemini response: {e}")
                        continue
                else:
                    detail = response.text[:200]
                    log.warning(f"Gemini model {model} returned {response.status_code}: {detail}, trying next fallback...")
                    last_error = RuntimeError(f"Gemini API error {response.status_code}: {detail}")

        if last_error:
            raise last_error
        raise RuntimeError("All Gemini candidate models failed")

    async def _generate_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        api_key = self.settings.openai_api_key or self.settings.llm_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or LLM_API_KEY not configured")

        url = "https://api.openai.com/v1/chat/completions"
        model = self.settings.openai_model

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

        with Timer("OpenAI LLM generation") as t:
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.llm_timeout
                ) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                    )
            except httpx.TimeoutException:
                raise ConnectionError("OpenAI API request timed out")

        if response.status_code >= 400:
            detail = response.text[:300]
            raise RuntimeError(f"OpenAI API error {response.status_code}: {detail}")

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse OpenAI response: {e}")

        log.info(f"OpenAI generated {len(text)} chars ({t.elapsed_ms:.0f}ms)")
        return text.strip()

    async def translate_to_english(self, text: str) -> str:
        """Translate a non-English query into English for cross-lingual vector retrieval."""
        prompt = (
            f"Translate the following user query into a concise English search query for retrieval. "
            f"Output ONLY the plain English translation, without explanation or quotes:\n\n{text}"
        )
        try:
            if self.settings.llm_provider == "gemini":
                translated = await self._generate_gemini(prompt)
            elif self.settings.llm_provider == "openai":
                translated = await self._generate_openai(prompt)
            else:
                translated = text
            return translated.strip().strip('"').strip("'")
        except Exception as e:
            log.warning(f"Cross-lingual query translation failed: {e}")
            return text

    async def translate_sources_to_language(
        self, texts: list[str], target_language: str
    ) -> list[str]:
        """Translate retrieved source snippets into the user's language in parallel."""
        if not texts or not target_language or target_language.lower() == "english":
            return texts

        import asyncio

        async def _translate_single(snippet: str) -> str:
            prompt = (
                f"Translate the following text passage into fluent {target_language}. "
                f"Output ONLY the direct plain translation without quotes or notes:\n\n{snippet[:400]}"
            )
            try:
                if self.settings.llm_provider == "gemini":
                    res = await self._generate_gemini(prompt)
                elif self.settings.llm_provider == "openai":
                    res = await self._generate_openai(prompt)
                else:
                    return snippet
                return res.strip().strip('"').strip("'")
            except Exception as e:
                log.warning(f"Single snippet translation fallback: {e}")
                return snippet

        try:
            return await asyncio.gather(*[_translate_single(t) for t in texts])
        except Exception as e:
            log.warning(f"Source snippet batch translation failed: {e}")
            return texts


