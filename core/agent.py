"""
Agent orchestration for the Health Assistant.

Uses a LangGraph StateGraph to implement the ReAct loop with agentic RAG:
agent_node (LLM) → should_continue? → tool_node → grade_retrieval → agent_node

The ``grade_retrieval`` node evaluates knowledge-base results and guides
the agent to try alternative strategies when retrieval is not relevant.
"""

import logging
import operator
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from core.prompts import SYSTEM_PROMPT, RETRIEVAL_GRADING_PROMPT

logger = logging.getLogger(__name__)


# ── State Schema ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


# ── Graph Construction ───────────────────────────────────────────────────
def build_graph(
    llm: ChatOpenAI,
    tools: list[BaseTool],
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the agent graph.

    Args:
        llm: The ChatOpenAI instance.
        tools: List of LangChain tools for the agent.
        checkpointer: Optional LangGraph checkpointer for persistent
            memory.  When provided, the graph saves and restores state
            automatically using a ``thread_id`` passed at invoke time.

    Returns:
        A compiled LangGraph runnable.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Call the LLM with the current conversation and system prompt."""
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)]
            + state["messages"]
        )
        # Extract token usage from response metadata.
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "messages": [response],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    def should_continue(state: AgentState) -> str:
        """Route to tools if the LLM requested tool calls, else end."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "end"

    def grade_retrieval(state: AgentState) -> dict:
        """Evaluate knowledge-base results and guide the agent if needed.

        Runs after the tools node.  If ``search_knowledge_base`` was
        called, a lightweight LLM classification determines whether the
        retrieved documents are relevant.  When they are not, a guidance
        message is appended so the agent can try an alternative strategy
        (e.g. web search).

        If no knowledge-base search occurred, the node passes through
        without any state change.
        """
        messages = state["messages"]

        # Find the most recent AIMessage with tool calls.
        last_ai: AIMessage | None = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_ai = msg
                break

        if last_ai is None:
            return {}

        # Check if search_knowledge_base was among the tool calls.
        kb_call_id: str | None = None
        for tc in last_ai.tool_calls:
            if tc["name"] == "search_knowledge_base":
                kb_call_id = tc["id"]
                break

        if kb_call_id is None:
            return {}

        # Find the corresponding ToolMessage with the retrieval results.
        kb_result: str = ""
        for msg in reversed(messages):
            if (
                isinstance(msg, ToolMessage)
                and msg.tool_call_id == kb_call_id
            ):
                kb_result = msg.content
                break

        if not kb_result or "No relevant information" in kb_result:
            return {}

        # Find the user's original question (last HumanMessage).
        user_question: str = ""
        for msg in reversed(messages):
            if msg.type == "human":
                user_question = msg.content[:500]
                break

        # Grade the retrieval with a lightweight LLM call.
        grade_prompt = RETRIEVAL_GRADING_PROMPT.format(
            question=user_question,
            documents=kb_result[:1000],
        )
        grade_response = llm.invoke(grade_prompt)
        grade = grade_response.content.strip().upper()
        logger.info("Retrieval grading result: '%s'", grade)

        if grade.startswith("YES"):
            return {}

        # Not relevant — add guidance for the agent's next iteration.
        logger.info("Knowledge base results graded as not relevant")
        guidance = SystemMessage(
            content=(
                "The knowledge base results were not relevant to the "
                "user's question. Consider using the search_web tool "
                "or answering from your general knowledge instead."
            )
        )
        return {"messages": [guidance]}

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("grade_retrieval", grade_retrieval)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph_builder.add_edge("tools", "grade_retrieval")
    graph_builder.add_edge("grade_retrieval", "agent")

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info(
        "Agent graph compiled with %d tools: %s",
        len(tools),
        ", ".join(t.name for t in tools),
    )
    return graph
