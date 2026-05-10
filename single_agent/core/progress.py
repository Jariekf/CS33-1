"""Runtime progress logging helpers."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def log_step(name: str) -> Iterator[None]:
    """Print start/end progress messages with elapsed wall-clock time."""

    start = perf_counter()
    print(f"[START] {name}", flush=True)
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start
        print(f"[FAILED] {name} ({elapsed:.2f}s)", flush=True)
        raise
    else:
        elapsed = perf_counter() - start
        print(f"[DONE]  {name} ({elapsed:.2f}s)", flush=True)
