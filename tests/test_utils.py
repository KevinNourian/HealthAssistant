"""Tests for core.utils — message parsing, thread IDs, and cost calculation."""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.utils import (
    build_thread_id,
    calculate_session_cost,
    extract_tools_and_sources,
)


# ═══════════════════════════════════════════════════════════════════════════════
# extract_tools_and_sources
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractToolsAndSources:
    """Verify tool and source extraction from message sequences."""

    def test_no_messages(self) -> None:
        tools, sources = extract_tools_and_sources([])
        assert tools == []
        assert sources == []

    def test_human_message_only(self) -> None:
        msgs = [HumanMessage(content="What is diabetes?")]
        tools, sources = extract_tools_and_sources(msgs)
        assert tools == []
        assert sources == []

    def test_ai_with_tool_calls(self, ai_message_with_tool_calls) -> None:
        tools, sources = extract_tools_and_sources([ai_message_with_tool_calls])
        assert tools == ["search_knowledge_base", "search_web"]

    def test_no_duplicate_tools(self) -> None:
        # Same tool called twice — should appear only once.
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_web", "args": {"query": "q1"}, "id": "c1"},
                {"name": "search_web", "args": {"query": "q2"}, "id": "c2"},
            ],
        )
        tools, _ = extract_tools_and_sources([ai_msg])
        assert tools == ["search_web"]

    def test_extracts_urls_from_tool_message(
        self, ai_message_with_tool_calls, tool_message_with_urls
    ) -> None:
        msgs = [ai_message_with_tool_calls, tool_message_with_urls]
        _, sources = extract_tools_and_sources(msgs)
        assert "https://www.cdc.gov/diabetes" in sources
        assert "https://www.mayo.org/diabetes-treatment" in sources

    def test_knowledge_base_fallback_source(
        self, ai_message_with_tool_calls, tool_message_no_urls
    ) -> None:
        # AI called search_knowledge_base, ToolMessage has no URLs.
        # Only include the KB tool call (not search_web).
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_knowledge_base", "args": {"query": "diabetes"}, "id": "call_1"},
            ],
        )
        msgs = [ai_msg, tool_message_no_urls]
        tools, sources = extract_tools_and_sources(msgs)
        assert "search_knowledge_base" in tools
        assert sources == ["Knowledge Base"]

    def test_web_urls_take_precedence_over_kb_fallback(
        self, ai_message_with_tool_calls, tool_message_with_urls, tool_message_no_urls
    ) -> None:
        # Both tools called, but web search has URLs — no KB fallback needed.
        msgs = [
            ai_message_with_tool_calls,
            tool_message_no_urls,
            tool_message_with_urls,
        ]
        tools, sources = extract_tools_and_sources(msgs)
        assert "search_knowledge_base" in tools
        assert "Knowledge Base" not in sources
        assert len(sources) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# build_thread_id
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildThreadId:
    """Thread ID construction from username and counter."""

    def test_basic_format(self) -> None:
        assert build_thread_id("alice", 0) == "alice_0"

    def test_incremented_counter(self) -> None:
        assert build_thread_id("bob", 3) == "bob_3"


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_session_cost
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateSessionCost:
    """Token usage and cost calculation."""

    def test_empty_history(self) -> None:
        result = calculate_session_cost([], "gpt-4o-mini")
        assert result.total_input == 0
        assert result.total_output == 0
        assert result.total_cost == 0.0

    def test_single_turn_cost(self) -> None:
        history = [{"input_tokens": 1_000_000, "output_tokens": 1_000_000}]
        result = calculate_session_cost(history, "gpt-4o-mini")
        # gpt-4o-mini: $0.15/1M input + $0.60/1M output
        assert result.total_input == 1_000_000
        assert result.total_output == 1_000_000
        assert result.total_cost == pytest.approx(0.75)

    def test_multiple_turns_accumulated(self) -> None:
        history = [
            {"input_tokens": 500, "output_tokens": 200},
            {"input_tokens": 300, "output_tokens": 100},
        ]
        result = calculate_session_cost(history, "gpt-4o-mini")
        assert result.total_input == 800
        assert result.total_output == 300

    def test_unknown_model_defaults_to_mini(self) -> None:
        history = [{"input_tokens": 1_000_000, "output_tokens": 1_000_000}]
        result = calculate_session_cost(history, "unknown-model-xyz")
        # Should fall back to gpt-4o-mini pricing.
        assert result.total_cost == pytest.approx(0.75)
