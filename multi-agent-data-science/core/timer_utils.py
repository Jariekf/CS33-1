"""Lightweight timer node factory for sub-graph agents."""

from __future__ import annotations

import time
from typing import Callable

from core.global_state import GlobalState

_COL = 28  # label column width for aligned terminal output


def make_timer_start(key: str) -> Callable[[GlobalState], GlobalState]:
    """Return a node function that stamps timing[key]['start'] = now."""
    def _node(state: GlobalState) -> GlobalState:
        timing = dict(state.get("timing") or {})
        timing[key] = {"start": time.time()}
        return {"timing": timing}
    _node.__name__ = f"_timer_start_{key}"
    return _node


def make_timer_end(key: str) -> Callable[[GlobalState], GlobalState]:
    """Return a node function that stamps timing[key]['end'], computes elapsed, prints it."""
    def _node(state: GlobalState) -> GlobalState:
        timing = dict(state.get("timing") or {})
        entry = dict(timing.get(key) or {})
        now = time.time()
        entry["end"] = now
        elapsed = now - entry.get("start", now)
        print(f"[timer] {key:<{_COL}} {elapsed:6.1f}s")
        timing[key] = entry
        return {"timing": timing}
    _node.__name__ = f"_timer_end_{key}"
    return _node
