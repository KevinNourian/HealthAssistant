"""
Per-user sliding-window rate limiter for the Health Assistant.

Persists request timestamps in a SQLite table so that rate-limit
state survives server restarts.  Call ``init_rate_limiter`` once at
startup to point the module at a database file; if not initialised,
the limiter falls back to an in-memory dict (original behaviour).
"""

import logging
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

# ── In-memory fallback (used when SQLite is not initialised) ─────────
_request_log: dict[str, list[float]] = {}

# ── SQLite state (set by init_rate_limiter) ──────────────────────────
_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


class RateLimitExceeded(Exception):
    """Raised when a user has exceeded their request quota."""


# ── Initialisation ───────────────────────────────────────────────────
def init_rate_limiter(db_path: str = "checkpoints.db") -> None:
    """Open a SQLite connection and create the rate-limits table.

    Safe to call more than once — subsequent calls are no-ops.
    Uses ``check_same_thread=False`` because Streamlit runs
    callbacks on different threads.

    Args:
        db_path: Path to the SQLite database file.
    """
    global _conn  # noqa: PLW0603

    if _conn is not None:
        return

    _conn = sqlite3.connect(db_path, check_same_thread=False)
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_limits (
            username  TEXT    NOT NULL,
            timestamp REAL    NOT NULL
        )
        """
    )
    _conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rate_limits_user
        ON rate_limits (username)
        """
    )
    _conn.commit()
    logger.info("Rate limiter initialised with SQLite: %s", db_path)


# ── Core logic ───────────────────────────────────────────────────────
def check_and_record(
    username: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """Check whether *username* is within the rate limit and record the request.

    When SQLite is initialised (via ``init_rate_limiter``), timestamps
    are persisted so limits survive server restarts.  Otherwise falls
    back to an in-memory dict.

    Args:
        username: The authenticated username to throttle.
        max_requests: Maximum number of requests allowed in *window_seconds*.
        window_seconds: Length of the sliding window in seconds.

    Raises:
        RateLimitExceeded: If the user has already reached *max_requests*
            within the current window.  The exception message includes the
            number of seconds until the oldest request ages out.
    """
    if _conn is not None:
        _check_and_record_sqlite(username, max_requests, window_seconds)
    else:
        _check_and_record_memory(username, max_requests, window_seconds)


# ── SQLite-backed implementation ─────────────────────────────────────
def _check_and_record_sqlite(
    username: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """Persist-backed rate limiting using the SQLite ``rate_limits`` table."""
    now = time.time()
    cutoff = now - window_seconds

    with _lock:
        # Prune expired rows for this user.
        _conn.execute(
            "DELETE FROM rate_limits WHERE username = ? AND timestamp <= ?",
            (username, cutoff),
        )

        # Count remaining (active) timestamps.
        row = _conn.execute(
            "SELECT COUNT(*) FROM rate_limits WHERE username = ?",
            (username,),
        ).fetchone()
        count = row[0]

        if count >= max_requests:
            # Find the oldest timestamp to compute wait time.
            oldest_row = _conn.execute(
                "SELECT MIN(timestamp) FROM rate_limits WHERE username = ?",
                (username,),
            ).fetchone()
            oldest = oldest_row[0]
            _conn.commit()
            _raise_limit_exceeded(
                username, count, max_requests, window_seconds, oldest, cutoff,
            )

        # Record the new request.
        _conn.execute(
            "INSERT INTO rate_limits (username, timestamp) VALUES (?, ?)",
            (username, now),
        )
        _conn.commit()

    logger.info(
        "Request recorded for user '%s' (%d/%d in window)",
        username,
        count + 1,
        max_requests,
    )


# ── In-memory fallback implementation ────────────────────────────────
def _check_and_record_memory(
    username: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    """In-memory rate limiting (original behaviour, not restart-safe)."""
    now = time.time()
    cutoff = now - window_seconds

    timestamps = _request_log.get(username, [])
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= max_requests:
        oldest = min(timestamps)
        _raise_limit_exceeded(
            username, len(timestamps), max_requests,
            window_seconds, oldest, cutoff,
        )

    timestamps.append(now)
    _request_log[username] = timestamps
    logger.info(
        "Request recorded for user '%s' (%d/%d in window)",
        username,
        len(timestamps),
        max_requests,
    )


# ── Shared helper ────────────────────────────────────────────────────
def _raise_limit_exceeded(
    username: str,
    count: int,
    max_requests: int,
    window_seconds: int,
    oldest: float,
    cutoff: float,
) -> None:
    """Build and raise a ``RateLimitExceeded`` with a human-readable wait time."""
    seconds_until_reset = int(oldest - cutoff) + 1
    minutes, seconds = divmod(seconds_until_reset, 60)
    wait_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    logger.warning(
        "Rate limit exceeded for user '%s' (%d/%d requests in window)",
        username,
        count,
        max_requests,
    )
    raise RateLimitExceeded(
        f"You've reached the limit of {max_requests} requests per "
        f"{window_seconds // 60} minute(s). "
        f"Please wait {wait_str} before trying again."
    )
