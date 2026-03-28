"""
Per-user sliding-window rate limiter for the Health Assistant.

Tracks request timestamps in a module-level dict keyed by username.
State persists across Streamlit reruns (module state is shared within
a single server process) and resets harmlessly on server restart.
"""

import logging
import time

logger = logging.getLogger(__name__)

# username -> list of UNIX timestamps of recent requests
_request_log: dict[str, list[float]] = {}


class RateLimitExceeded(Exception):
    """Raised when a user has exceeded their request quota."""


def check_and_record(
    username: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """Check whether *username* is within the rate limit and record the request.

    Prunes timestamps older than *window_seconds*, checks the remaining
    count against *max_requests*, then appends the current timestamp.

    Args:
        username: The authenticated username to throttle.
        max_requests: Maximum number of requests allowed in *window_seconds*.
        window_seconds: Length of the sliding window in seconds.

    Raises:
        RateLimitExceeded: If the user has already reached *max_requests*
            within the current window.  The exception message includes the
            number of seconds until the oldest request ages out.
    """
    now = time.monotonic()
    cutoff = now - window_seconds

    timestamps = _request_log.get(username, [])
    # Prune expired timestamps
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= max_requests:
        oldest = min(timestamps)
        seconds_until_reset = int(oldest - cutoff) + 1
        minutes, seconds = divmod(seconds_until_reset, 60)
        if minutes:
            wait_str = f"{minutes}m {seconds}s"
        else:
            wait_str = f"{seconds}s"
        logger.warning(
            "Rate limit exceeded for user '%s' (%d/%d requests in window)",
            username,
            len(timestamps),
            max_requests,
        )
        raise RateLimitExceeded(
            f"You've reached the limit of {max_requests} requests per "
            f"{window_seconds // 60} minute(s). "
            f"Please wait {wait_str} before trying again."
        )

    timestamps.append(now)
    _request_log[username] = timestamps
    logger.info(
        "Request recorded for user '%s' (%d/%d in window)",
        username,
        len(timestamps),
        max_requests,
    )
