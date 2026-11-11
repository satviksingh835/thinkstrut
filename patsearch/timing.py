"""Minimal per-stage stopwatch used by the build pipeline, the search engine and the benchmark."""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Generator


class Stopwatch:
    """Accumulates wall-clock milliseconds per named stage.

        sw = Stopwatch()
        with sw.stage("embed_query"):
            ...                      # timed work
        sw.as_dict()                 # {"embed_query": 6.1, "total": 6.1}

    Entering the same stage twice adds to its total. Search responses return these numbers as
    `timings_ms`; the CLI footer, the web UI's timing bar and the benchmark all read them.
    """

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        # perf_counter_ns is monotonic with nanosecond resolution — needed for sub-millisecond stages
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter_ns() - t0) / 1e6

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return sum(self.ms.values())

    def as_dict(self, digits: int = 3) -> dict[str, float]:
        out = {k: round(v, digits) for k, v in self.ms.items()}
        out["total"] = round(self.total_ms, digits)
        return out
