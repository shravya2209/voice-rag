"""RAG orchestration pipeline — structured, typed, stage-by-stage execution."""

from __future__ import annotations

import re
import time
import uuid

from app.config import get_settings
from app.data.models import RetrievedChunk, RAGResponse
from app.audio.elevenlabs_stt import SpeechToTextProvider
from app.embeddings.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.llm import LLMProvider
from app.guardrails.safety import check_safety
from app.guardrails.relevance import check_retrieval_confidence, check_query_relevance
from app.guardrails.grounding import check_grounding
from app.utils.logging import get_logger

log = get_logger("orchestration.pipeline")


class RAGPipeline:
    """End-to-end RAG pipeline with typed stages and timing.

    Pipeline stages:
    1. Receive request
    2. Validate input
    3. Speech-to-text (if audio)
    4. Query preprocessing
    5. Guardrail pre-check (safety + relevance)
    6. Embedding
    7. Retrieval
    8. Reranking (if enabled)
    9. Retrieval confidence check
    10. Context construction
    11. LLM generation
    12. Grounding validation
    13. Response formatting
    14. Metrics/logging
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.stt = SpeechToTextProvider()
        self.embedder = Embedder.get_instance()
        self.retriever = HybridRetriever()
        self.reranker = Reranker() if self.settings.use_reranker else None
        self.llm = LLMProvider()

    async def process_text_query(self, query: str) -> RAGResponse:
        """Process a text query through the full RAG pipeline."""
        request_id = str(uuid.uuid4())[:8]
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        log.info(f"[{request_id}] Processing text query: '{query[:80]}'")

        # ── Stage 2: Validate input ───────────────────────────────
        query = self._preprocess_query(query)

        # ── Stage 5: Guardrail pre-check ──────────────────────────
        if self.settings.guardrails_enabled:
            safety = check_safety(query)
            if not safety["safe"]:
                return RAGResponse(
                    query=query,
                    answer=safety["reason"],
                    guardrail_flags={"safety": safety},
                    timings={"total": self._elapsed_ms(t_start)},
                )

            relevance = check_query_relevance(query)
            if not relevance["relevant"]:
                return RAGResponse(
                    query=query,
                    answer=relevance["reason"],
                    guardrail_flags={"relevance": relevance},
                    timings={"total": self._elapsed_ms(t_start)},
                )

        # ── Multilingual / Cross-lingual Query Handling ───────────
        retrieval_query = query
        if any(ord(c) > 127 for c in query):
            try:
                retrieval_query = await self.llm.translate_to_english(query)
                log.info(f"[{request_id}] Cross-lingual retrieval query: '{query}' -> '{retrieval_query}'")
            except Exception as e:
                log.warning(f"[{request_id}] Query translation fallback: {e}")
                retrieval_query = query

        # ── Stage 6: Embedding ────────────────────────────────────
        t0 = time.perf_counter()
        _query_emb = self.embedder.embed_text(retrieval_query)
        timings["embedding"] = self._elapsed_ms(t0)

        # ── Stage 7: Retrieval ────────────────────────────────────
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(
            retrieval_query,
            top_k=self.settings.retrieval_candidates,
        )
        timings["retrieval"] = self._elapsed_ms(t0)

        # ── Stage 8: Reranking ────────────────────────────────────
        if self.reranker and candidates:
            t0 = time.perf_counter()
            candidates = self.reranker.rerank(
                retrieval_query, candidates, top_k=self.settings.top_k
            )
            timings["reranking"] = self._elapsed_ms(t0)
        else:
            candidates = candidates[: self.settings.top_k]

        # ── Stage 9: Retrieval confidence check ───────────────────
        if self.settings.guardrails_enabled:
            confidence = check_retrieval_confidence(candidates)
            if not confidence["confident"]:
                return RAGResponse(
                    query=query,
                    answer=confidence["reason"],
                    sources=candidates,
                    guardrail_flags={"confidence": confidence},
                    timings={**timings, "total": self._elapsed_ms(t_start)},
                )

        # ── Stage 10: Context construction ────────────────────────
        context_passages = [
            {
                "text": rc.chunk.text,
                "score": rc.rerank_score if rc.rerank_score > 0 else rc.fused_score,
                "chunk_id": rc.chunk.chunk_id,
            }
            for rc in candidates
        ]

        # ── Stage 11: LLM generation ─────────────────────────────
        t0 = time.perf_counter()
        try:
            answer = await self.llm.generate(query, context_passages)
        except Exception as e:
            log.error(f"[{request_id}] LLM generation failed: {e}")
            answer = (
                "I encountered an error generating the answer. "
                "Please try again."
            )
        timings["generation"] = self._elapsed_ms(t0)

        # ── Stage 12: Grounding validation ────────────────────────
        grounding_result = {"grounded": True, "score": 1.0}
        if self.settings.guardrails_enabled and candidates:
            context_texts = [rc.chunk.text for rc in candidates]
            grounding_result = check_grounding(answer, context_texts)

        # ── Stage 12.5: Dynamic Source Translation for Display ────
        from app.generation.prompts import _detect_language
        detected_lang = _detect_language(query)
        if detected_lang != "English" and candidates:
            try:
                raw_texts = [rc.chunk.text for rc in candidates]
                translated_texts = await self.llm.translate_sources_to_language(raw_texts, detected_lang)
                translated_candidates = []
                for idx, rc in enumerate(candidates):
                    trans_text = translated_texts[idx] if idx < len(translated_texts) else rc.chunk.text
                    translated_chunk = rc.chunk.model_copy(update={"text": trans_text, "language": detected_lang})
                    translated_candidates.append(rc.model_copy(update={"chunk": translated_chunk}))
                candidates = translated_candidates
            except Exception as e:
                log.warning(f"[{request_id}] Source translation fallback: {e}")

        # ── Stage 13 & 14: Response + metrics ─────────────────────
        total_ms = self._elapsed_ms(t_start)
        timings["total"] = total_ms

        log.info(
            f"[{request_id}] Complete: {len(candidates)} sources, "
            f"grounding={grounding_result['score']:.2f}, "
            f"total={total_ms:.0f}ms"
        )

        return RAGResponse(
            query=query,
            answer=answer,
            sources=candidates,
            grounding_score=grounding_result["score"],
            guardrail_flags={"grounding": grounding_result},
            timings=timings,
        )

    async def process_voice_query(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> tuple[str, RAGResponse]:
        """Process a voice query: STT → RAG pipeline.

        Args:
            audio_bytes: Raw audio data
            filename: Audio filename
            language: Language code (e.g. 'kn', 'hi', 'en', 'auto')

        Returns:
            Tuple of (transcript, RAGResponse)
        """
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        # ── Stage 3: Speech-to-text ───────────────────────────────
        t0 = time.perf_counter()
        transcript = await self.stt.transcribe(
            audio_bytes, filename=filename, language=language
        )
        timings["stt"] = self._elapsed_ms(t0)

        # ── Validate transcription ────────────────────────────────
        if not transcript or len(transcript.strip()) < 2:
            response = RAGResponse(
                query="",
                answer="Could not understand the audio. Please try again.",
                timings={**timings, "total": self._elapsed_ms(t_start)},
            )
            return "", response

        # ── Run text pipeline ─────────────────────────────────────
        response = await self.process_text_query(transcript)

        # Merge STT timing
        response.timings["stt"] = timings["stt"]
        response.timings["total"] = self._elapsed_ms(t_start)

        return transcript, response

    def _preprocess_query(self, query: str) -> str:
        """Normalize and clean the query text."""
        query = query.strip()
        # Collapse whitespace
        query = re.sub(r"\s+", " ", query)
        return query

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000
