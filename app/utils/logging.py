import io
import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a consistently-formatted logger with safe UTF-8 output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        out_stream = sys.stdout
        if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
            out_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        handler = logging.StreamHandler(out_stream)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)-25s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

