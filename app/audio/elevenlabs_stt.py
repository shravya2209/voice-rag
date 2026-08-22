"""ElevenLabs Speech-to-Text provider with full error handling."""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.timing import Timer

log = get_logger("audio.elevenlabs")

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


class SpeechToTextProvider:
    """ElevenLabs STT with structured error handling."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._validate_config()

    def _validate_config(self) -> None:
        if not self.settings.elevenlabs_api_key:
            log.warning("ELEVENLABS_API_KEY not set - STT will fail at runtime")

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str | None = None,
    ) -> str:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data
            filename: Filename with extension for MIME detection
            language: Language code (e.g. 'kn', 'hi', 'en', or 'auto'/None for auto-detection)

        Returns:
            Transcribed text

        Raises:
            ValueError: If audio is empty or too large
            ConnectionError: If ElevenLabs API is unreachable
            RuntimeError: For API errors
        """
        # ── Validate input ─────────────────────────────────────────
        if not audio_bytes:
            raise ValueError("Empty audio data")

        max_bytes = self.settings.max_audio_size_mb * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise ValueError(
                f"Audio too large: {len(audio_bytes)} bytes "
                f"(max {self.settings.max_audio_size_mb} MB)"
            )

        if not self.settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")

        # ── Call ElevenLabs API ─────────────────────────────────────
        data_payload: dict[str, str] = {
            "model_id": self.settings.elevenlabs_stt_model,
        }
        # Only pass language_code if a specific code is requested (not auto or None)
        if language and language.strip() and language.strip().lower() not in ("auto", "none"):
            data_payload["language_code"] = language.strip().lower()

        with Timer("ElevenLabs STT") as t:
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.elevenlabs_timeout
                ) as client:
                    response = await client.post(
                        ELEVENLABS_STT_URL,
                        headers={
                            "xi-api-key": self.settings.elevenlabs_api_key
                        },
                        files={"file": (filename, audio_bytes)},
                        data=data_payload,
                    )
            except httpx.TimeoutException:
                raise ConnectionError("ElevenLabs STT request timed out")
            except httpx.ConnectError:
                raise ConnectionError("Could not connect to ElevenLabs API")

        # ── Handle response ────────────────────────────────────────
        if response.status_code == 429:
            raise RuntimeError("ElevenLabs rate limit exceeded - try again later")

        if response.status_code == 401:
            raise RuntimeError("Invalid ElevenLabs API key")

        if response.status_code >= 400:
            detail = response.text[:200]
            raise RuntimeError(
                f"ElevenLabs STT error {response.status_code}: {detail}"
            )

        try:
            result = response.json()
        except Exception:
            raise RuntimeError("Malformed response from ElevenLabs API")

        text = result.get("text", "").strip()
        if not text:
            raise ValueError("Transcription returned empty text")

        log.info(
            f"Transcribed ({t.elapsed_ms:.0f}ms): "
            f"'{text[:80]}{'...' if len(text) > 80 else ''}'"
        )
        return text


class TextToSpeechProvider:
    """ElevenLabs TTS with error handling."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> bytes:
        """Convert text to speech audio bytes (mp3).

        Args:
            text: Text to synthesize (max 5000 chars)
            voice_id: Optional voice override

        Returns:
            Audio bytes in mp3 format
        """
        if not text or not text.strip():
            raise ValueError("Empty text for TTS")

        if not self.settings.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")

        vid = voice_id or self.settings.elevenlabs_tts_voice
        url = f"{ELEVENLABS_TTS_URL}/{vid}"

        with Timer("ElevenLabs TTS") as t:
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.elevenlabs_timeout
                ) as client:
                    response = await client.post(
                        url,
                        headers={
                            "xi-api-key": self.settings.elevenlabs_api_key,
                            "Content-Type": "application/json",
                        },
                        json={
                            "text": text[:5000],
                            "model_id": self.settings.elevenlabs_tts_model,
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75,
                            },
                        },
                    )
            except httpx.TimeoutException:
                raise ConnectionError("ElevenLabs TTS request timed out")
            except httpx.ConnectError:
                raise ConnectionError("Could not connect to ElevenLabs API")

        if response.status_code >= 400:
            raise RuntimeError(
                f"ElevenLabs TTS error {response.status_code}: {response.text[:200]}"
            )

        log.info(f"TTS synthesized {len(response.content)} bytes ({t.elapsed_ms:.0f}ms)")
        return response.content
