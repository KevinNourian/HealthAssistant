"""
Agent orchestration for the Health Assistant.

Binds tools to the LLM and implements the agent loop that lets the model
decide which tools to call and when to produce a final answer.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from prompts import SYSTEM_PROMPT


MAX_AGENT_ITERATIONS = 5


def bind_tools(llm: ChatOpenAI, tools: list):
    """Bind tools to the LLM and return the bound model plus a name→tool map.

    Args:
        llm: The ChatOpenAI instance.
        tools: List of LangChain tool objects.

    Returns:
        A tuple of (llm_with_tools, tools_by_name).
    """
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    return llm_with_tools, tools_by_name


def run_agent(
    user_input: str,
    messages: list,
    llm_with_tools,
    tools_by_name: dict,
) -> str:
    """Run the agent loop.

    The LLM decides which tools to call and when to produce a final answer.
    Messages are mutated in-place so the caller can inspect tool calls
    after the function returns.

    Args:
        user_input: The user's message (may include attached PDF text).
        messages: The LangChain message list (mutated in-place).
        llm_with_tools: The LLM with tools bound.
        tools_by_name: A dict mapping tool names to tool objects.

    Returns:
        The assistant's final text answer.
    """
    messages.append(HumanMessage(content=user_input))

    for _ in range(MAX_AGENT_ITERATIONS):
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        )
        messages.append(response)

        # No tool calls → final answer
        if not response.tool_calls:
            return response.content

        # Execute each tool call and feed results back
        for tool_call in response.tool_calls:
            tool_fn = tools_by_name[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    # Safety: return whatever we have if the iteration limit is reached
    return response.content or (
        "I wasn't able to complete the request. Please try again."
    )
