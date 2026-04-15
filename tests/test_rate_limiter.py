"""Tests for core.rate_limiter — in-memory and SQLite rate limiting."""

import time

import pytest

import core.rate_limiter as rl_module
from core.rate_limiter import (
    RateLimitExceeded,
    _request_log,
    check_and_record,
    init_rate_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter_state():
    """Reset both in-memory and SQLite state before each test.

    The rate limiter module uses a global ``_conn`` for SQLite and
    ``_request_log`` for in-memory mode.  We reset both so tests
    don't leak state into each other.
    """
    _request_log.clear()
    # Force _conn back to None so each test starts in in-memory mode
    # unless it explicitly calls init_rate_limiter().
    rl_module._conn = None
    yield
    # Teardown: close any SQLite connection opened during the test.
    if rl_module._conn is not None:
        rl_module._conn.close()
        rl_module._conn = None
    _request_log.clear()


class TestCheckAndRecord:
    """Core rate-limiting behaviour using the in-memory fallback."""

    def test_first_request_allowed(self) -> None:
        # The very first request for a user should always succeed.
        check_and_record("alice", max_requests=5, window_seconds=60)

    def test_requests_up_to_limit_allowed(self) -> None:
        # Exactly max_requests calls should all succeed.
        for _ in range(5):
            check_and_record("alice", max_requests=5, window_seconds=60)

    def test_exceeding_limit_raises(self) -> None:
        # The (max_requests + 1)th call should raise.
        for _ in range(3):
            check_and_record("alice", max_requests=3, window_seconds=60)

        with pytest.raises(RateLimitExceeded):
            check_and_record("alice", max_requests=3, window_seconds=60)

    def test_different_users_independent(self) -> None:
        # Alice hitting her limit should not affect Bob.
        for _ in range(3):
            check_and_record("alice", max_requests=3, window_seconds=60)

        # Alice is now at the limit, but Bob should be fine.
        check_and_record("bob", max_requests=3, window_seconds=60)

    def test_window_expiry_allows_new_requests(self) -> None:
        # Use a tiny window so old timestamps expire quickly.
        check_and_record("alice", max_requests=1, window_seconds=1)

        # Immediately, Alice is at the limit.
        with pytest.raises(RateLimitExceeded):
            check_and_record("alice", max_requests=1, window_seconds=1)

        # After the window expires, she can make another request.
        time.sleep(1.1)
        check_and_record("alice", max_requests=1, window_seconds=1)

    def test_error_message_includes_limit_info(self) -> None:
        check_and_record("alice", max_requests=1, window_seconds=120)

        with pytest.raises(RateLimitExceeded, match=r"1 requests per 2 minute"):
            check_and_record("alice", max_requests=1, window_seconds=120)

    def test_boundary_exactly_at_limit(self) -> None:
        # Make exactly max_requests calls — all should pass.
        for _ in range(10):
            check_and_record("alice", max_requests=10, window_seconds=60)

        # The 11th should fail.
        with pytest.raises(RateLimitExceeded):
            check_and_record("alice", max_requests=10, window_seconds=60)


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite-backed rate limiting (production path)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckAndRecordSQLite:
    """Rate-limiting behaviour using the SQLite backend.

    Each test calls ``init_rate_limiter`` with a temp database via
    ``tmp_path``.  The ``reset_rate_limiter_state`` fixture (autouse)
    resets ``_conn`` to None before each test, so ``init_rate_limiter``
    creates a fresh connection every time.
    """

    def test_init_creates_table(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        init_rate_limiter(db_path)
        # Verify the table exists by querying it.
        row = rl_module._conn.execute(
            "SELECT COUNT(*) FROM rate_limits"
        ).fetchone()
        assert row[0] == 0

    def test_first_request_allowed(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        check_and_record("alice", max_requests=5, window_seconds=60)

    def test_requests_up_to_limit_allowed(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        for _ in range(5):
            check_and_record("alice", max_requests=5, window_seconds=60)

    def test_exceeding_limit_raises(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        for _ in range(3):
            check_and_record("alice", max_requests=3, window_seconds=60)

        with pytest.raises(RateLimitExceeded):
            check_and_record("alice", max_requests=3, window_seconds=60)

    def test_different_users_independent(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        for _ in range(3):
            check_and_record("alice", max_requests=3, window_seconds=60)

        # Alice is at the limit, but Bob should be fine.
        check_and_record("bob", max_requests=3, window_seconds=60)

    def test_window_expiry_allows_new_requests(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        check_and_record("alice", max_requests=1, window_seconds=1)

        with pytest.raises(RateLimitExceeded):
            check_and_record("alice", max_requests=1, window_seconds=1)

        time.sleep(1.1)
        check_and_record("alice", max_requests=1, window_seconds=1)

    def test_timestamps_persisted_in_db(self, tmp_path) -> None:
        init_rate_limiter(str(tmp_path / "test.db"))
        check_and_record("alice", max_requests=5, window_seconds=60)
        check_and_record("alice", max_requests=5, window_seconds=60)

        # Verify rows were written to the database.
        row = rl_module._conn.execute(
            "SELECT COUNT(*) FROM rate_limits WHERE username = ?",
            ("alice",),
        ).fetchone()
        assert row[0] == 2

    def test_init_is_idempotent(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        init_rate_limiter(db_path)
        conn_first = rl_module._conn

        # Calling again should be a no-op (same connection).
        init_rate_limiter(db_path)
        assert rl_module._conn is conn_first
