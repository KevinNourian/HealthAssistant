"""End-to-end tests for the agent graph with a mocked LLM.

These tests build a real LangGraph ``StateGraph`` via ``build_graph()``
but replace the LLM with fakes that return predetermined ``AIMessage``
objects.  This verifies that the graph wiring — agent → tools →
grade_retrieval → agent — works correctly without making any API calls.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from core.agent import build_graph


# ── Fake LLM classes ────────────────────────────────────────────────────
# These need to satisfy two interfaces used by build_graph:
#   1. llm.bind_tools(tools)  → returns an object whose .invoke()
#      returns AIMessage (used by the agent node)
#   2. llm.invoke(prompt)     → returns an object with .content
#      (used by the retrieval grader)


class _FakeResponse:
    """Mimics the response from llm.invoke() in the grading node."""

    def __init__(self, content: str) -> None:
        self.content = content


class _DirectAnswerBoundLLM:
    """Bound LLM that always responds directly (no tool calls).

    This simulates the simplest path through the graph:
    agent is called once, returns an answer, ``should_continue``
    routes to END.
    """

    def invoke(self, messages: list) -> AIMessage:
        return AIMessage(
            content="Diabetes is a chronic metabolic condition.",
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )


class _DirectAnswerLLM:
    """Fake LLM for the direct-answer test scenario."""

    def bind_tools(self, tools: list) -> _DirectAnswerBoundLLM:
        return _DirectAnswerBoundLLM()

    def invoke(self, prompt: str) -> _FakeResponse:
        # Grading node — shouldn't be reached in the direct path,
        # but return YES as a safe default.
        return _FakeResponse("YES")


class _ToolCallingBoundLLM:
    """Bound LLM that calls a tool on the first turn, then answers.

    First call:  returns an AIMessage with a tool_call for
                 ``search_knowledge_base``.
    Second call: returns a plain AIMessage with the final answer
                 (no tool calls), so ``should_continue`` routes to END.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def invoke(self, messages: list) -> AIMessage:
        self._call_count += 1
        if self._call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "diabetes"},
                        "id": "call_1",
                    }
                ],
                usage_metadata={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
            )
        return AIMessage(
            content="Based on the knowledge base: diabetes is a chronic condition.",
            usage_metadata={"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
        )


class _ToolCallingLLM:
    """Fake LLM for the tool-calling test scenario."""

    def bind_tools(self, tools: list) -> _ToolCallingBoundLLM:
        return _ToolCallingBoundLLM()

    def invoke(self, prompt: str) -> _FakeResponse:
        # Called by grade_retrieval to evaluate relevance.
        return _FakeResponse("YES")


# ── Test tools ──────────────────────────────────────────────────────────
# Minimal real tools so that LangGraph's ToolNode can execute them.

@tool
def search_knowledge_base(query: str) -> str:
    """Search the health knowledge base."""
    return "Diabetes is a metabolic disorder affecting blood sugar regulation."


@tool
def search_web(query: str) -> str:
    """Search the web for health information."""
    return "Web result: diabetes overview."


# ── Tests ───────────────────────────────────────────────────────────────

class TestAgentGraphDirectAnswer:
    """LLM responds directly without calling any tools."""

    def test_returns_answer(self) -> None:
        graph = build_graph(
            llm=_DirectAnswerLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        assert "Diabetes" in result["messages"][-1].content

    def test_tracks_token_usage(self) -> None:
        graph = build_graph(
            llm=_DirectAnswerLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 20

    def test_no_tool_calls_in_response(self) -> None:
        graph = build_graph(
            llm=_DirectAnswerLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert not last_msg.tool_calls


class TestAgentGraphToolCalling:
    """LLM calls a tool, then generates a final answer."""

    def test_tool_result_in_messages(self) -> None:
        graph = build_graph(
            llm=_ToolCallingLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        # The final message should be the LLM's answer after seeing tool results.
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "knowledge base" in last_msg.content.lower()

    def test_accumulates_tokens_across_turns(self) -> None:
        graph = build_graph(
            llm=_ToolCallingLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        # First call: 40 input + 10 output
        # Second call: 80 input + 30 output
        # Total: 120 input, 40 output
        assert result["input_tokens"] == 120
        assert result["output_tokens"] == 40

    def test_message_sequence(self) -> None:
        graph = build_graph(
            llm=_ToolCallingLLM(),
            tools=[search_knowledge_base, search_web],
            checkpointer=None,
            max_context_messages=10,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is diabetes?")]}
        )
        messages = result["messages"]
        # Expected sequence: HumanMessage → AIMessage (tool call) →
        # ToolMessage (result) → AIMessage (final answer)
        # (grade_retrieval may insert a SystemMessage if grading fails,
        #  but our grader returns YES so no extra message.)
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[1].tool_calls  # first AI message requested a tool
        assert isinstance(messages[-1], AIMessage)
        assert not messages[-1].tool_calls  # final message is the answer
