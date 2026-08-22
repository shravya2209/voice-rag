"""Latency measurement utilities for benchmarking pipeline components."""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """Statistics for a set of latency measurements."""
    component: str
    values_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values_ms)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values_ms) if self.values_ms else 0.0

    @property
    def min(self) -> float:
        return min(self.values_ms) if self.values_ms else 0.0

    @property
    def max(self) -> float:
        return max(self.values_ms) if self.values_ms else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p70(self) -> float:
        return self._percentile(70)

    @property
    def p100(self) -> float:
        return self.max

    def _percentile(self, p: int) -> float:
        if not self.values_ms:
            return 0.0
        sorted_vals = sorted(self.values_ms)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "count": self.count,
            "mean": round(self.mean, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p50": round(self.p50, 2),
            "p70": round(self.p70, 2),
            "p100": round(self.p100, 2),
        }


class LatencyTracker:
    """Collects latency measurements across multiple runs."""

    def __init__(self) -> None:
        self._stats: dict[str, LatencyStats] = {}

    def record(self, component: str, ms: float) -> None:
        if component not in self._stats:
            self._stats[component] = LatencyStats(component=component)
        self._stats[component].values_ms.append(ms)

    def record_from_timings(self, timings: dict[str, float]) -> None:
        """Record all component timings from a pipeline run."""
        for component, ms in timings.items():
            self.record(component, ms)

    def get_stats(self) -> dict[str, LatencyStats]:
        return dict(self._stats)

    def print_table(self) -> str:
        """Format results as an ASCII table."""
        header = f"{'Component':<15} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8} {'Min':>8} {'Max':>8}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for name, stats in sorted(self._stats.items()):
            lines.append(
                f"{name:<15} {stats.p50:>7.1f}ms {stats.p70:>7.1f}ms "
                f"{stats.p100:>7.1f}ms {stats.mean:>7.1f}ms "
                f"{stats.min:>7.1f}ms {stats.max:>7.1f}ms"
            )

        lines.append(sep)
        table = "\n".join(lines)
        return table

    def to_dict_list(self) -> list[dict]:
        return [s.to_dict() for s in self._stats.values()]
