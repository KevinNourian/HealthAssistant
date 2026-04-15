"""Shared fixtures for the Health Assistant test suite."""

import pytest

from langchain_core.messages import AIMessage, ToolMessage


# ── Mock LLM ────────────────────────────────────────────────────────────
class MockResponse:
    """Mimics the object returned by ChatOpenAI.invoke()."""

    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    """Fake LLM that returns a predetermined response.

    Implements the same `.invoke(prompt)` interface that
    ``is_health_related`` expects from a real ChatOpenAI instance.
    """

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> MockResponse:
        self.last_prompt = prompt
        return MockResponse(self.response_text)


class ErrorLLM:
    """Fake LLM that always raises an exception.

    Used to verify that ``is_health_related`` fails open.
    """

    def invoke(self, prompt: str) -> None:
        raise RuntimeError("API connection failed")


@pytest.fixture
def mock_llm_yes() -> MockLLM:
    """LLM that always responds 'YES'."""
    return MockLLM("YES")


@pytest.fixture
def mock_llm_no() -> MockLLM:
    """LLM that always responds 'NO'."""
    return MockLLM("NO")


@pytest.fixture
def error_llm() -> ErrorLLM:
    """LLM that always raises an exception."""
    return ErrorLLM()


# ── Message Fixtures ────────────────────────────────────────────────────
@pytest.fixture
def ai_message_with_tool_calls() -> AIMessage:
    """An AIMessage that requested two tool calls."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_knowledge_base", "args": {"query": "diabetes"}, "id": "call_1"},
            {"name": "search_web", "args": {"query": "diabetes treatment"}, "id": "call_2"},
        ],
    )


@pytest.fixture
def tool_message_with_urls() -> ToolMessage:
    """A ToolMessage containing web search results with URL: lines."""
    return ToolMessage(
        content=(
            "**Diabetes Overview**\n"
            "Diabetes is a chronic condition.\n"
            "URL: https://www.cdc.gov/diabetes\n\n"
            "**Treatment Options**\n"
            "Treatment includes lifestyle changes.\n"
            "URL: https://www.mayo.org/diabetes-treatment"
        ),
        tool_call_id="call_2",
    )


@pytest.fixture
def tool_message_no_urls() -> ToolMessage:
    """A ToolMessage from the knowledge base (no URLs)."""
    return ToolMessage(
        content="Diabetes is a metabolic disorder affecting blood sugar levels.",
        tool_call_id="call_1",
    )
