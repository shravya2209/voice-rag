"""Aggregated metrics collection for the application."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AppMetrics:
    """Global application metrics."""
    total_queries: int = 0
    total_voice_queries: int = 0
    total_text_queries: int = 0
    total_errors: int = 0
    guardrail_blocks: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_query(self, latency_ms: float, is_voice: bool = False) -> None:
        self.total_queries += 1
        if is_voice:
            self.total_voice_queries += 1
        else:
            self.total_text_queries += 1
        self._latencies.append(latency_ms)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_error(self) -> None:
        self.total_errors += 1

    def record_guardrail_block(self) -> None:
        self.guardrail_blocks += 1

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_voice_queries": self.total_voice_queries,
            "total_text_queries": self.total_text_queries,
            "total_errors": self.total_errors,
            "guardrail_blocks": self.guardrail_blocks,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


# Global singleton
metrics = AppMetrics()
