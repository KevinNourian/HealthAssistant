"""Integration tests for core.utils.process_query.

``process_query`` is the main entry point that chains:
crisis detection → rate limiting → health topic check → graph invocation.

These tests use mock graph and LLM objects to verify the orchestration
logic without making any API calls.
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from core.guardrails import CrisisType
from core.rate_limiter import _request_log
from core.utils import process_query

from tests.conftest import MockLLM


# ── Test configuration ──────────────────────────────────────────────────
_TEST_CONFIG: dict = {
    "rate_limit": {"max_requests": 100, "window_seconds": 60},
}


# ── Mock graph ──────────────────────────────────────────────────────────
class _MockStateSnapshot:
    """Mimics the object returned by graph.get_state()."""

    def __init__(self, values: dict | None = None) -> None:
        self.values = values or {}


class _MockGraph:
    """Fake LangGraph runnable that returns a predetermined response.

    Implements ``get_state()`` and ``invoke()`` — the two methods
    that ``process_query`` calls on the graph object.
    """

    def __init__(self, answer: str = "This is the agent's response.") -> None:
        self.answer = answer
        self.last_invoke_input: dict | None = None

    def get_state(self, config: dict) -> _MockStateSnapshot:
        return _MockStateSnapshot({"messages": [], "input_tokens": 0, "output_tokens": 0})

    def invoke(self, inputs: dict, config: dict) -> dict:
        self.last_invoke_input = inputs
        return {
            "messages": [
                inputs["messages"][0],
                AIMessage(
                    content=self.answer,
                    usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                ),
            ],
            "input_tokens": 100,
            "output_tokens": 50,
        }


class _ErrorGraph(_MockGraph):
    """Fake graph that raises an exception on invoke."""

    def invoke(self, inputs: dict, config: dict) -> dict:
        raise RuntimeError("LLM provider is down")


# ── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def clear_rate_limiter():
    """Reset in-memory rate limiter between tests."""
    _request_log.clear()
    yield
    _request_log.clear()


# ── Tests ───────────────────────────────────────────────────────────────

class TestProcessQueryCrisisPath:
    """Crisis detection should short-circuit before the graph is invoked."""

    def test_crisis_returns_crisis_result(self) -> None:
        result = process_query(
            graph=_MockGraph(),
            llm=MockLLM("YES"),
            question="I want to kill myself",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert result.crisis is not None
        assert result.crisis.detected is True
        assert result.crisis.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS

    def test_crisis_skips_graph_invocation(self) -> None:
        graph = _MockGraph()
        process_query(
            graph=graph,
            llm=MockLLM("YES"),
            question="I want to kill myself",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        # graph.invoke() should never have been called.
        assert graph.last_invoke_input is None

    def test_crisis_does_not_count_toward_rate_limit(self) -> None:
        process_query(
            graph=_MockGraph(),
            llm=MockLLM("YES"),
            question="I want to kill myself",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        # No entry in the rate limiter because crisis short-circuited.
        assert "alice" not in _request_log


class TestProcessQueryRateLimitPath:
    """Rate limiting should block after exceeding the quota."""

    def test_rate_limit_exceeded_returns_error(self) -> None:
        config = {"rate_limit": {"max_requests": 1, "window_seconds": 60}}
        # First call succeeds.
        process_query(
            graph=_MockGraph(),
            llm=MockLLM("YES"),
            question="What is diabetes?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=config,
        )
        # Second call should be rate-limited.
        result = process_query(
            graph=_MockGraph(),
            llm=MockLLM("YES"),
            question="What is hypertension?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=config,
        )
        assert result.error is not None
        assert "limit" in result.error.lower()
        assert result.success is False


class TestProcessQueryOutOfScope:
    """Out-of-scope queries should be rejected by the health topic classifier."""

    def test_non_health_query_rejected(self) -> None:
        result = process_query(
            graph=_MockGraph(),
            llm=MockLLM("NO"),
            question="What is the capital of France?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert result.out_of_scope is True
        assert result.success is False

    def test_out_of_scope_skips_graph_invocation(self) -> None:
        graph = _MockGraph()
        process_query(
            graph=graph,
            llm=MockLLM("NO"),
            question="What is the capital of France?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert graph.last_invoke_input is None


class TestProcessQuerySuccessPath:
    """Happy path: health query passes all guardrails and reaches the graph."""

    def test_successful_query(self) -> None:
        result = process_query(
            graph=_MockGraph(answer="Diabetes is a chronic condition."),
            llm=MockLLM("YES"),
            question="What is diabetes?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert result.success is True
        assert "Diabetes" in result.answer
        assert result.error is None
        assert result.crisis is None
        assert result.out_of_scope is False

    def test_token_counts_populated(self) -> None:
        result = process_query(
            graph=_MockGraph(),
            llm=MockLLM("YES"),
            question="What is diabetes?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_graph_error_returns_error_result(self) -> None:
        result = process_query(
            graph=_ErrorGraph(),
            llm=MockLLM("YES"),
            question="What is diabetes?",
            uploaded_lab=None,
            username="alice",
            chat_thread_counter=0,
            config=_TEST_CONFIG,
        )
        assert result.success is False
        assert result.error is not None
        assert "down" in result.error.lower()
