"""In-process sliding-window rate limiter (per client key), used by the HTTP API."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable


class RateLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = self.clock()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= 60.0:
                events.popleft()
            if len(events) >= self.per_minute:
                return False
            events.append(now)
            return True
