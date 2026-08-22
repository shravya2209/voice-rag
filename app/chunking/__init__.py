"""Chunking module — multiple strategies selectable via config."""

from app.chunking.base import BaseChunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.metadata import MetadataChunker


_STRATEGY_MAP: dict[str, type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "semantic": SemanticChunker,
    "metadata": MetadataChunker,
}


def get_chunker(strategy: str | None = None) -> BaseChunker:
    """Factory: return a chunker instance by strategy name.

    Args:
        strategy: One of 'fixed', 'sentence', 'semantic', 'metadata'.
                  If None, reads from config CHUNKING_STRATEGY.
    """
    if strategy is None:
        from app.config import get_settings
        strategy = get_settings().chunking_strategy

    cls = _STRATEGY_MAP.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Available: {list(_STRATEGY_MAP.keys())}"
        )
    return cls()


__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "MetadataChunker",
    "get_chunker",
]
