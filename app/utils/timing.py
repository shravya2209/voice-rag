"""Performance timing utilities — critical for latency optimization."""

from __future__ import annotations

import time
import functools
from typing import Callable, Any

from app.utils.logging import get_logger

log = get_logger("timing")


def timed(func: Callable) -> Callable:
    """Decorator that logs execution time of a function."""

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(f"{func.__qualname__}: {elapsed:.1f}ms")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        log.info(f"{func.__qualname__}: {elapsed:.1f}ms")
        return result

    if asyncio_iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is an async coroutine."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.label:
            log.info(f"{self.label}: {self.elapsed_ms:.1f}ms")
