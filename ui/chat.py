"""Chat tab UI — agent-powered health Q&A."""

import logging
from typing import Any

import streamlit as st

from core.config import SOURCE_KNOWLEDGE_BASE
from core.guardrails import CrisisType
from core.user_data import save_user_data
from core.utils import QueryResult, process_query_streaming

logger = logging.getLogger(__name__)


def render(username: str, graph: Any, llm: Any, config: dict[str, Any]) -> None:
    """Render the Chat tab.

    Args:
        username: The authenticated username.
        graph: The compiled LangGraph agent.
        llm: The LLM instance.
        config: The application configuration dictionary.
    """
    st.markdown("### Health Chat")
    st.caption(
        "Ask questions, request document summaries, or upload a lab report "
        "for analysis — the AI decides which tool to use."
    )

    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        for qa in st.session_state.conversation_history:

            st.markdown(
                f"""
                <div class="chat-question">
                    <strong style="color: #1F2937;">Q:</strong>
                    {qa['question']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="chat-answer">
                    <strong style="color: #1F2937;">A:</strong><br><br>
                    {qa['answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if qa.get("tools_used"):
                st.caption(f"🔧 Tools used: {', '.join(qa['tools_used'])}")

            if qa.get("input_tokens") or qa.get("output_tokens"):
                st.caption(
                    f"📊 Tokens: {qa.get('input_tokens', 0):,} in "
                    f"/ {qa.get('output_tokens', 0):,} out"
                )

            if qa.get("sources"):
                for i, source in enumerate(qa["sources"], 1):
                    if source == SOURCE_KNOWLEDGE_BASE:
                        st.caption(f"📄 Source: {SOURCE_KNOWLEDGE_BASE}")
                    else:
                        st.caption(f"🔗 Source {i}: {source}")

        st.markdown("---")

    # Input area
    question: str = st.text_input(
        "Question",
        placeholder=(
            "e.g. What does my knowledge base say about blood pressure?"
        ),
        key=f"health_question_{st.session_state.question_counter}",
        label_visibility="collapsed",
    )

    uploaded_lab = st.file_uploader(
        "Attach a lab report (optional)",
        type=["pdf"],
        key=f"chat_upload_{st.session_state.question_counter}",
    )

    col_ask, col_clear = st.columns(2)

    with col_ask:
        ask_button: bool = st.button(
            "Ask", key="ask_btn", use_container_width=True,
        )

    with col_clear:
        if st.button(
            "Clear Chat", key="clear_btn", use_container_width=True,
        ):
            st.session_state.conversation_history = []
            st.session_state.chat_thread_counter += 1
            st.session_state.question_counter += 1
            save_user_data(username)
            logger.info("Chat history cleared by user")
            st.rerun()

    if ask_button:
        if question:
            # Placeholders for real-time streaming feedback.
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            accumulated_text = ""
            result = None

            for event in process_query_streaming(
                graph=graph,
                llm=llm,
                question=question,
                uploaded_lab=uploaded_lab,
                username=username,
                chat_thread_counter=st.session_state.chat_thread_counter,
                config=config,
            ):
                if event.event_type == "token":
                    accumulated_text += event.content
                    response_placeholder.markdown(accumulated_text + "▌")
                elif event.event_type == "tool_start":
                    status_placeholder.info(
                        f"🔧 Using **{event.content}**..."
                    )
                elif event.event_type == "complete":
                    result = event.result
                elif event.event_type == "crisis":
                    result = event.result
                elif event.event_type == "out_of_scope":
                    result = QueryResult(out_of_scope=True)
                elif event.event_type == "error":
                    result = QueryResult(error=event.content)

            # Clear streaming placeholders before final render.
            status_placeholder.empty()
            response_placeholder.empty()

            # ── Display result ────────────────────────────────────────
            if result is None:
                st.error("No response received. Please try again.")
                st.session_state.question_counter += 1
            elif result.crisis:
                if result.crisis.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS:
                    st.error(
                        "It sounds like you may be going through a very "
                        "difficult time. Please reach out for support "
                        "right away — you don't have to face this alone."
                    )
                else:
                    st.error(
                        "This sounds like it could be a medical emergency. "
                        "Please seek immediate help."
                    )
                for resource in result.crisis.resources:
                    st.markdown(resource)
                st.session_state.question_counter += 1
            elif result.out_of_scope:
                st.session_state.conversation_history.append({
                    "question": question,
                    "answer": (
                        "This assistant is designed for health and "
                        "wellness questions only. Please ask a "
                        "health-related question."
                    ),
                    "tools_used": [],
                    "sources": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
                st.session_state.question_counter += 1
                st.rerun()
            elif result.error:
                st.error(f"Error: {result.error}")
                st.session_state.question_counter += 1
            elif result.success:
                st.session_state.conversation_history.append({
                    "question": question,
                    "answer": result.answer,
                    "tools_used": result.tools_used,
                    "sources": result.sources,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                })
                st.session_state.question_counter += 1
                st.rerun()
        else:
            st.warning("Please enter a question")
