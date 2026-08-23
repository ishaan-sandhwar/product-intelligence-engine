"""Request pacing for free-tier providers.

Gemini's free tier allows 15 requests per minute per model, and a rejected
request still consumes one of them. The limiter has to hold the caller back
rather than let a burst poison the following window.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.provider import _RateLimiter, _retry_after  # noqa: E402


def test_calls_within_the_limit_do_not_block():
    limiter = _RateLimiter(5)
    started = time.perf_counter()
    for _ in range(5):
        limiter.acquire()
    assert time.perf_counter() - started < 0.5


def test_the_call_over_the_limit_waits_for_the_window():
    """The 3rd call into a 2-per-minute budget must wait, not sail through."""
    limiter = _RateLimiter(2)
    limiter._WINDOW_SECONDS = 0.6          # shrink the window to keep the test quick
    limiter.acquire()
    limiter.acquire()

    started = time.perf_counter()
    limiter.acquire()
    waited = time.perf_counter() - started
    assert waited >= 0.4


def test_zero_rpm_disables_pacing():
    """Providers without a declared ceiling are not throttled."""
    limiter = _RateLimiter(0)
    started = time.perf_counter()
    for _ in range(50):
        limiter.acquire()
    assert time.perf_counter() - started < 0.2


def test_limiter_is_thread_safe():
    """Concurrent workers share one budget; the window must not be overspent."""
    limiter = _RateLimiter(4)
    granted: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        limiter.acquire()
        with lock:
            granted.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(granted) == 4
    assert len(limiter._calls) == 4


def test_server_retry_hint_is_parsed():
    """Both shapes a 429 uses to state its backoff."""
    assert _retry_after("Please retry in 59.995708891s.") == 59.995708891
    assert _retry_after("'retryDelay': '59s'") == 59.0
    assert _retry_after("no hint at all") is None
