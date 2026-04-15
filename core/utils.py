"""
Utility functions for the Health Assistant.

Extracts business logic from the UI layer so that ``app.py`` remains
focused on display and user interaction.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from core.config import (
    MODEL_PRICING,
    SOURCE_KNOWLEDGE_BASE,
    SOURCE_URL_PREFIX,
    TOOL_SEARCH_KB,
)
from core.guardrails import check_for_crisis, is_health_related, CrisisResult
from core.rate_limiter import check_and_record, RateLimitExceeded

logger = logging.getLogger(__name__)


# ── Result Types ─────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    """Result of processing a user query through the agent pipeline."""

    success: bool = False
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    crisis: CrisisResult | None = None
    out_of_scope: bool = False
    error: str | None = None


@dataclass
class CostSummary:
    """Cumulative token usage and cost for a session."""

    total_input: int = 0
    total_output: int = 0
    total_cost: float = 0.0


@dataclass
class StreamEvent:
    """A single event yielded during streaming agent execution.

    Attributes:
        event_type: One of ``"token"``, ``"tool_start"``,
            ``"complete"``, ``"crisis"``, ``"out_of_scope"``,
            or ``"error"``.
        content: Token text for ``"token"`` events, tool name for
            ``"tool_start"``, or error message for ``"error"``.
        result: The final ``QueryResult``, populated only on
            ``"complete"`` and ``"crisis"`` events.
    """

    event_type: str
    content: str = ""
    result: QueryResult | None = None


# ── Thread ID ────────────────────────────────────────────────────────────
def build_thread_id(username: str, counter: int) -> str:
    """Construct a checkpoint thread ID from username and counter.

    Args:
        username: The authenticated username.
        counter: The chat thread counter (incremented on Clear Chat).

    Returns:
        A string like ``"alice_0"`` or ``"bob_3"``.
    """
    return f"{username}_{counter}"


# ── Conversation Context ─────────────────────────────────────────────────
def get_recent_context(graph: Any, thread_id: str, max_messages: int = 4) -> str:
    """Extract recent conversation context from the checkpoint.

    Used to give the health topic classifier context for follow-up
    queries like "explain that in simpler terms."

    Args:
        graph: The compiled LangGraph runnable.
        thread_id: The checkpoint thread ID.
        max_messages: Maximum number of recent messages to include.

    Returns:
        A formatted string of recent messages, or empty string if
        no history exists.
    """
    try:
        state_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        prev_msgs = state_snapshot.values.get("messages", [])
        context = ""
        for msg in prev_msgs[-max_messages:]:
            role = msg.type
            context += f"{role}: {msg.content[:200]}\n"
        return context
    except Exception as e:
        logger.error("Failed to get recent context: %s", e)
        return ""


# ── PDF Extraction ───────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    """Extract text content from an uploaded PDF file.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` object.

    Returns:
        The concatenated text from all pages, or an empty string
        if extraction fails.
    """
    try:
        reader = PdfReader(BytesIO(uploaded_file.getbuffer()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        logger.error("PDF text extraction failed: %s", e)
        return ""


# ── Message Parsing ──────────────────────────────────────────────────────
def extract_tools_and_sources(
    messages: list[BaseMessage],
) -> tuple[list[str], list[str]]:
    """Identify tool names and source URLs from a list of messages.

    Scans for ``AIMessage`` objects with tool calls and
    ``ToolMessage`` objects containing ``URL:`` lines.  If no web
    sources are found but ``search_knowledge_base`` was used, the
    source list defaults to ``["Knowledge Base"]``.

    Args:
        messages: The messages to scan (typically the new messages
            from a single agent turn).

    Returns:
        A tuple of ``(tools_used, sources)``.
    """
    tools_used: list[str] = []
    sources: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])
        if isinstance(msg, ToolMessage) and SOURCE_URL_PREFIX in msg.content:
            for line in msg.content.split("\n"):
                if line.strip().startswith(SOURCE_URL_PREFIX):
                    url = line.strip().replace(SOURCE_URL_PREFIX, "").strip()
                    if url:
                        sources.append(url)

    if not sources and TOOL_SEARCH_KB in tools_used:
        sources = [SOURCE_KNOWLEDGE_BASE]

    return tools_used, sources


# ── Query Processing ─────────────────────────────────────────────────────
def process_query(
    graph: Any,
    llm: Any,
    question: str,
    uploaded_lab: Any | None,
    username: str,
    chat_thread_counter: int,
    config: dict[str, Any],
) -> QueryResult:
    """Process a user query through the full agent pipeline.

    Runs guardrails, rate limiting, optional PDF extraction, graph
    invocation, and post-processing.  Returns a ``QueryResult`` that
    ``app.py`` can use to render the appropriate UI response.

    Args:
        graph: The compiled LangGraph runnable.
        llm: The ChatOpenAI instance (for the health topic classifier).
        question: The user's question text.
        uploaded_lab: Optional uploaded lab report PDF file.
        username: The authenticated username.
        chat_thread_counter: Current chat thread counter.
        config: Application configuration dictionary.

    Returns:
        A ``QueryResult`` describing the outcome.
    """
    result = QueryResult()

    # ── Guardrail 1: Crisis detection (regex, no LLM) ────────────────
    crisis = check_for_crisis(question)
    if crisis.detected:
        result.crisis = crisis
        return result

    # ── Rate limit check ─────────────────────────────────────────────
    try:
        check_and_record(
            username,
            config["rate_limit"]["max_requests"],
            config["rate_limit"]["window_seconds"],
        )
    except RateLimitExceeded as e:
        result.error = str(e)
        return result

    # ── Guardrail 2: Health topic scope (LLM check) ──────────────────
    # Skip when a lab PDF is attached — context is clear.
    thread_id = build_thread_id(username, chat_thread_counter)

    if not uploaded_lab:
        recent_context = get_recent_context(graph, thread_id)
        if not is_health_related(question, llm, recent_context=recent_context):
            result.out_of_scope = True
            logger.info("Query rejected by topic scope guardrail")
            return result

    # ── PDF extraction (if lab report attached) ──────────────────────
    user_message: str = question
    if uploaded_lab:
        logger.info(
            "Extracting text from uploaded PDF: %s",
            uploaded_lab.name,
        )
        pdf_text = extract_pdf_text(uploaded_lab)
        if pdf_text.strip():
            user_message = f"{question}\n\n[Attached lab report text]\n{pdf_text}"
        else:
            result.error = (
                "Could not extract text from the uploaded PDF. "
                "The file may be image-based."
            )
            logger.warning("PDF text extraction returned empty content")
            return result

    # ── Graph invocation ─────────────────────────────────────────────
    try:
        invoke_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10,
        }

        # Snapshot state before invoke for token deltas.
        state_before = graph.get_state(invoke_config)
        prev_values = state_before.values or {}
        msgs_before: int = len(prev_values.get("messages", []))
        tokens_in_before: int = prev_values.get("input_tokens", 0)
        tokens_out_before: int = prev_values.get("output_tokens", 0)

        graph_result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            invoke_config,
        )

        result.success = True
        result.answer = graph_result["messages"][-1].content

        # Per-turn token counts (delta).
        result.input_tokens = graph_result.get("input_tokens", 0) - tokens_in_before
        result.output_tokens = graph_result.get("output_tokens", 0) - tokens_out_before

        # Extract tools and sources from new messages.
        new_messages = graph_result["messages"][msgs_before:]
        result.tools_used, result.sources = extract_tools_and_sources(new_messages)

    except GraphRecursionError:
        logger.warning("Agent hit recursion limit")
        result.error = "I wasn't able to complete the request. Please try again."
    except Exception as e:
        logger.error("Error during agent invocation: %s", e)
        result.error = str(e)

    return result


# ── Streaming Query Processing ───────────────────────────────────────────
def process_query_streaming(
    graph: Any,
    llm: Any,
    question: str,
    uploaded_lab: Any | None,
    username: str,
    chat_thread_counter: int,
    config: dict[str, Any],
) -> Generator[StreamEvent, None, None]:
    """Process a user query with token-level streaming.

    Runs the same guardrail and pre-processing pipeline as
    ``process_query``, then streams graph execution via
    ``graph.stream(stream_mode="messages")``.  Yields
    ``StreamEvent`` objects that the UI layer can consume to
    display partial responses in real time.

    Args:
        graph: The compiled LangGraph runnable.
        llm: The ChatOpenAI instance (for the health topic classifier).
        question: The user's question text.
        uploaded_lab: Optional uploaded lab report PDF file.
        username: The authenticated username.
        chat_thread_counter: Current chat thread counter.
        config: Application configuration dictionary.

    Yields:
        ``StreamEvent`` instances in this order:
        - Zero or more ``"tool_start"`` / ``"token"`` events
          during graph execution.
        - Exactly one terminal event: ``"complete"``,
          ``"crisis"``, ``"out_of_scope"``, or ``"error"``.
    """
    # ── Guardrail 1: Crisis detection (regex, no LLM) ────────────────
    crisis = check_for_crisis(question)
    if crisis.detected:
        yield StreamEvent("crisis", result=QueryResult(crisis=crisis))
        return

    # ── Rate limit check ─────────────────────────────────────────────
    try:
        check_and_record(
            username,
            config["rate_limit"]["max_requests"],
            config["rate_limit"]["window_seconds"],
        )
    except RateLimitExceeded as exc:
        yield StreamEvent("error", content=str(exc))
        return

    # ── Guardrail 2: Health topic scope (LLM check) ──────────────────
    thread_id = build_thread_id(username, chat_thread_counter)

    if not uploaded_lab:
        recent_context = get_recent_context(graph, thread_id)
        if not is_health_related(question, llm, recent_context=recent_context):
            logger.info("Query rejected by topic scope guardrail")
            yield StreamEvent("out_of_scope")
            return

    # ── PDF extraction (if lab report attached) ──────────────────────
    user_message: str = question
    if uploaded_lab:
        logger.info(
            "Extracting text from uploaded PDF: %s",
            uploaded_lab.name,
        )
        pdf_text = extract_pdf_text(uploaded_lab)
        if pdf_text.strip():
            user_message = (
                f"{question}\n\n[Attached lab report text]\n{pdf_text}"
            )
        else:
            yield StreamEvent(
                "error",
                content=(
                    "Could not extract text from the uploaded PDF. "
                    "The file may be image-based."
                ),
            )
            logger.warning("PDF text extraction returned empty content")
            return

    # ── Streaming graph invocation ───────────────────────────────────
    try:
        invoke_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10,
        }

        # Snapshot state before streaming for token deltas.
        state_before = graph.get_state(invoke_config)
        prev_values = state_before.values or {}
        msgs_before: int = len(prev_values.get("messages", []))
        tokens_in_before: int = prev_values.get("input_tokens", 0)
        tokens_out_before: int = prev_values.get("output_tokens", 0)

        tools_seen: set[str] = set()

        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_message)]},
            invoke_config,
            stream_mode="messages",
        ):
            node = metadata.get("langgraph_node", "")

            if node == "agent" and isinstance(chunk, AIMessageChunk):
                # Stream content tokens to the UI.
                if chunk.content:
                    yield StreamEvent("token", content=chunk.content)

                # Detect tool calls being generated.
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        name = tc.get("name", "")
                        if name and name not in tools_seen:
                            tools_seen.add(name)
                            yield StreamEvent("tool_start", content=name)

        # ── Post-stream: build the final QueryResult ─────────────────
        state_after = graph.get_state(invoke_config)
        after_values = state_after.values or {}

        result = QueryResult(success=True)
        result.answer = after_values["messages"][-1].content

        # Per-turn token counts (delta).
        result.input_tokens = (
            after_values.get("input_tokens", 0) - tokens_in_before
        )
        result.output_tokens = (
            after_values.get("output_tokens", 0) - tokens_out_before
        )

        # Extract tools and sources from new messages.
        new_messages = after_values.get("messages", [])[msgs_before:]
        result.tools_used, result.sources = extract_tools_and_sources(
            new_messages,
        )

        yield StreamEvent("complete", result=result)

    except GraphRecursionError:
        logger.warning("Agent hit recursion limit")
        yield StreamEvent(
            "error",
            content="I wasn't able to complete the request. Please try again.",
        )
    except Exception as exc:
        logger.error("Error during agent streaming: %s", exc)
        yield StreamEvent("error", content=str(exc))


# ── Cost Calculation ─────────────────────────────────────────────────────
def calculate_session_cost(
    conversation_history: list[dict[str, Any]],
    model_name: str,
) -> CostSummary:
    """Calculate cumulative token usage and cost for the session.

    Args:
        conversation_history: List of conversation turn dicts, each
            containing ``input_tokens`` and ``output_tokens``.
        model_name: The model name to look up in ``MODEL_PRICING``.

    Returns:
        A ``CostSummary`` with total tokens and cost.
    """
    total_input = sum(qa.get("input_tokens", 0) for qa in conversation_history)
    total_output = sum(qa.get("output_tokens", 0) for qa in conversation_history)

    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o-mini"])
    total_cost = (total_input / 1_000_000) * pricing["input"] + (
        total_output / 1_000_000
    ) * pricing["output"]

    return CostSummary(
        total_input=total_input,
        total_output=total_output,
        total_cost=total_cost,
    )
