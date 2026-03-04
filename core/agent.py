"""
Agent orchestration for the Health Assistant.

Binds tools to the LLM and implements the agent loop that lets the model
decide which tools to call and when to produce a final answer.  The loop
includes a configurable iteration limit to prevent runaway tool-call
chains.
"""

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from core.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS: int = 5
"""Maximum number of LLM tool round-trips before the loop stops."""


def bind_tools(
    llm: ChatOpenAI,
    tools: list[BaseTool],
) -> tuple[Any, dict[str, BaseTool]]:
    """Bind tools to the LLM and return the bound model plus a lookup map.

    Returns:
        A tuple of ``(llm_with_tools, tools_by_name)`` where
        *llm_with_tools* is the tool-augmented model and
        *tools_by_name* maps each tool's name to its callable.
    """
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    logger.info(
        "Bound %d tools to LLM: %s",
        len(tools),
        ", ".join(tools_by_name.keys()),
    )
    return llm_with_tools, tools_by_name


def run_agent(
    user_input: str,
    messages: list[BaseMessage],
    llm_with_tools: Any,
    tools_by_name: dict[str, BaseTool],
) -> str:
    """Run the agent loop.

    The LLM decides which tools to call and when to produce a final
    answer.  *messages* is mutated in-place so the caller can inspect
    tool calls and results after the function returns.

    Args:
        user_input: The user's message (may include attached PDF text).
        messages: The LangChain message list (mutated in-place).
        llm_with_tools: The LLM with tools bound via ``bind_tools``.
        tools_by_name: A dict mapping tool names to tool callables.

    Returns:
        The assistant's final text answer.

    Raises:
        No exceptions are raised; errors inside individual tools are
        caught by the tools themselves and returned as error strings.
    """
    logger.info("Agent invoked with input (%d chars)", len(user_input))
    messages.append(HumanMessage(content=user_input))

    response: AIMessage | None = None

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        logger.info(
            "Agent iteration %d/%d",
            iteration,
            MAX_AGENT_ITERATIONS,
        )

        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + messages
        )
        messages.append(response)

        # No tool calls — the LLM produced a final answer
        if not response.tool_calls:
            logger.info(
                "Agent produced final answer on iteration %d",
                iteration,
            )
            return response.content

        # Execute each tool call and feed results back
        for tool_call in response.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call["args"]
            logger.info(
                "Calling tool '%s' with args: %s",
                tool_name,
                tool_args,
            )

            if tool_name not in tools_by_name:
                error_msg = f"Unknown tool requested: {tool_name}"
                logger.error(error_msg)
                messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call["id"])
                )
                continue

            result: str = tools_by_name[tool_name].invoke(tool_args)
            logger.info(
                "Tool '%s' returned %d chars", tool_name, len(result)
            )
            messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    # Safety: return whatever we have if the iteration limit is reached
    logger.warning(
        "Agent hit max iterations (%d) without a final answer",
        MAX_AGENT_ITERATIONS,
    )
    if response is not None and response.content:
        return response.content
    return "I wasn't able to complete the request. Please try again."
