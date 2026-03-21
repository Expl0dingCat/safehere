"""Regex timeout wrapper to guard against ReDoS attacks."""

import concurrent.futures


class _TimeoutSentinel:
    """Sentinel returned when a regex search exceeds its time budget."""

    def __repr__(self):
        # type: () -> str
        return "TIMEOUT"


TIMEOUT = _TimeoutSentinel()

# Module-level executor reused across all calls.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def safe_search(pattern, text, timeout_ms=50):
    # type: (re.Pattern, str, int) -> Union[Optional[re.Match], _TimeoutSentinel]
    """Run *pattern.search(text)* with a wall-clock timeout.

    Returns the match object (or ``None``) when the search completes in time.
    Returns the :data:`TIMEOUT` sentinel when the deadline is exceeded.

    Works on both Windows and Unix (no ``signal.alarm``).
    """
    future = _executor.submit(pattern.search, text)
    try:
        return future.result(timeout=timeout_ms / 1000.0)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return TIMEOUT
