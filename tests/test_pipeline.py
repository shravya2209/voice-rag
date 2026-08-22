"""Integration test for the RAG pipeline using mocked external services."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.data.models import Chunk, RetrievedChunk, RAGResponse


class TestPipelineModels:
    def test_chunk_creation(self):
        c = Chunk(
            chunk_id="p0_c0",
            document_id="p0",
            text="Test chunk",
            strategy="sentence",
        )
        assert c.chunk_id == "p0_c0"
        assert c.strategy == "sentence"

    def test_retrieved_chunk(self):
        c = Chunk(chunk_id="p0_c0", document_id="p0", text="Test")
        rc = RetrievedChunk(chunk=c, dense_score=0.85, fused_score=0.85)
        assert rc.dense_score == 0.85

    def test_rag_response(self):
        r = RAGResponse(
            query="test",
            answer="test answer",
            timings={"total": 150.0},
        )
        assert r.query == "test"
        assert r.timings["total"] == 150.0


class TestPipelineLogic:
    """Tests pipeline stages without needing real APIs or index."""

    @pytest.mark.asyncio
    async def test_pipeline_with_mocked_services(self):
        """Test full pipeline with mocked STT, retriever, and LLM."""
        with patch("app.orchestration.pipeline.RAGPipeline.__init__", return_value=None):
            from app.orchestration.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)

            # Mock settings
            mock_settings = MagicMock()
            mock_settings.guardrails_enabled = True
            mock_settings.use_reranker = False
            mock_settings.top_k = 3
            mock_settings.retrieval_candidates = 10
            mock_settings.min_relevance_score = 0.2
            pipeline.settings = mock_settings

            # Mock embedder
            pipeline.embedder = MagicMock()
            pipeline.embedder.embed_text.return_value = [0.1] * 384

            # Mock retriever
            mock_chunks = [
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=f"c{i}",
                        document_id=f"d{i}",
                        text=f"Test passage {i} about science and technology.",
                    ),
                    dense_score=0.8 - i * 0.1,
                    fused_score=0.8 - i * 0.1,
                )
                for i in range(3)
            ]
            pipeline.retriever = MagicMock()
            pipeline.retriever.retrieve.return_value = mock_chunks

            pipeline.reranker = None

            # Mock LLM
            pipeline.llm = MagicMock()
            pipeline.llm.generate = AsyncMock(
                return_value="This is a test answer about science."
            )

            # Mock STT
            pipeline.stt = MagicMock()
            pipeline.stt.transcribe = AsyncMock(return_value="test query")

            result = await pipeline.process_text_query("What is science?")
            assert isinstance(result, RAGResponse)
            assert result.answer != ""
            assert len(result.sources) > 0
