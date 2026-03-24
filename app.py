"""
Health Assistant - Streamlit Application

Multi-user health assistant with a tool-calling agent, document management,
and health journaling.  Minimalist Nordic design.

This is the main entry point.  It wires together all modules and renders
the Streamlit UI.  Business logic lives in the other modules:

* ``core/prompts.py``   – LLM prompt templates
* ``core/config.py``    – configuration loading and constants
* ``core/auth.py``      – authentication setup
* ``core/user_data.py`` – user data persistence and session state
* ``core/tools.py``     – LangChain tool definitions
* ``core/agent.py``     – agent loop orchestration
"""

import logging
import os
import sqlite3
from io import BytesIO
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import load_config, save_config, validate_api_keys, MODEL_PRICING
from core.rate_limiter import check_and_record, RateLimitExceeded
from core.auth import load_authenticator, render_login
from core.user_data import init_session_state, save_user_data
from core.tools import create_tools
from core.agent import build_graph
from core.guardrails import check_for_crisis, is_health_related, CrisisType
from core.vector_store import (
    get_or_create_vectorstore,
    get_hybrid_retriever,
    add_pdf_to_vectorstore,
    remove_pdf_from_vectorstore,
    sync_vectorstore,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EXTERNAL STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
def load_css(path: str) -> None:
    """Inject an external CSS file into the Streamlit page.

    Args:
        path: Absolute or relative path to the ``.css`` file.
    """
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(os.path.join(os.path.dirname(__file__), "assets", "style.css"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()

try:
    validate_api_keys()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

config: dict[str, Any] = load_config()

# Keep a session-state copy so that adds/deletes are reflected immediately
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = list(config["pdf_files"])
config["pdf_files"] = st.session_state.pdf_files


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
authenticator = load_authenticator()

if (
    "authentication_status" not in st.session_state
    or st.session_state["authentication_status"] is not True
):
    render_login(authenticator)

# User is authenticated from this point onward
name: str = st.session_state["name"]
username: str = st.session_state["username"]
logger.info("Authenticated session for user '%s'", username)

init_session_state(username)

# Thread counter for checkpointed conversations.  Incrementing this
# starts a fresh conversation thread while preserving old checkpoints.
if "chat_thread_counter" not in st.session_state:
    st.session_state.chat_thread_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE RESOURCES (CACHED)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def initialize_vectorstore():
    """Initialize the vector store — runs once and caches the result.

    Returns:
        A tuple of ``(vectorstore, retriever)``.
    """
    logger.info("Initializing vector store")
    vectorstore = get_or_create_vectorstore(
        pdf_paths=config["pdf_files"],
        persist_directory=config["chroma_directory"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        force_recreate=False,
    )
    sync_vectorstore(
        vectorstore=vectorstore,
        pdf_paths=config["pdf_files"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    retriever = get_hybrid_retriever(vectorstore, k=config["retriever"]["k"])
    logger.info("Vector store ready")
    return vectorstore, retriever


@st.cache_resource
def initialize_llm() -> ChatOpenAI:
    """Initialize the LLM — runs once and caches the result.

    Returns:
        A configured ``ChatOpenAI`` instance.
    """
    logger.info(
        "Initializing LLM: model=%s, temperature=%s",
        config["llm"]["model"],
        config["llm"]["temperature"],
    )
    return ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
    )


@st.cache_resource
def initialize_checkpointer() -> SqliteSaver:
    """Initialize the SQLite checkpointer for persistent memory.

    Uses ``check_same_thread=False`` because Streamlit runs callbacks
    on different threads.

    Returns:
        A configured ``SqliteSaver`` instance.
    """
    logger.info("Initializing SQLite checkpointer")
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return checkpointer


with st.spinner("Loading..."):
    vectorstore, retriever = initialize_vectorstore()
    llm = initialize_llm()
    checkpointer = initialize_checkpointer()


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════
tools = create_tools(
    vectorstore,
    retriever,
    llm,
    config,
    SERPAPI_API_KEY,
    bp_readings_ref=st.session_state.bp_readings,
)
graph = build_graph(llm, tools, checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: SIDEBAR CONFIG ITEM
# ═══════════════════════════════════════════════════════════════════════════════
def render_config_item(label: str, value: Any) -> None:
    """Render a single configuration value in the sidebar.

    Args:
        label: Short human-readable label.
        value: The configuration value to display.
    """
    st.markdown(
        f"""
        <div class="config-item">
            <div class="config-label">{label}</div>
            <div class="config-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"**Welcome, {name}!**")

    try:
        authenticator.logout(location="sidebar", button_name="Sign Out")
    except TypeError:
        authenticator.logout("Sign Out", "sidebar")

    st.markdown("---")

    # ── Reminders ────────────────────────────────────────────────────────
    st.markdown("### ⏰ Reminders")

    with st.form("reminder_form", clear_on_submit=True):
        reminder_text = st.text_input(
            "Reminder",
            placeholder="e.g., Doctor visit",
            label_visibility="collapsed",
        )
        reminder_date = st.date_input("Date", label_visibility="collapsed")

        if st.form_submit_button("Add Reminder", use_container_width=True):
            if reminder_text:
                st.session_state.reminders.append({
                    "text": reminder_text,
                    "date": reminder_date.strftime("%Y-%m-%d"),
                    "id": len(st.session_state.reminders),
                })
                save_user_data(username)
                st.rerun()

    if st.session_state.reminders:
        for i, reminder in enumerate(st.session_state.reminders):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"""
                    <div style="padding: 8px 0;
                                border-bottom: 1px solid #E8E8E8;">
                        <span style="color: #6B7280; font-size: 12px;
                               font-weight: 500;">{reminder['date']}</span>
                        <br>
                        <span style="color: #1F2937;
                               font-size: 13px;">{reminder['text']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button(
                    "✕", key=f"del_rem_{i}", help="Delete",
                    use_container_width=True,
                ):
                    st.session_state.reminders.pop(i)
                    save_user_data(username)
                    st.rerun()
    else:
        st.caption("No reminders yet")

    st.markdown("---")

    # ── Knowledge Base ───────────────────────────────────────────────────
    st.markdown("### 📚 Documents")

    if config["pdf_files"]:
        for pdf in config["pdf_files"]:
            st.caption(f"• {os.path.basename(pdf)}")
    else:
        st.caption("No documents loaded")

    st.markdown("---")

    # ── Settings ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")

    col1, col2 = st.columns(2)
    with col1:
        render_config_item("Model", config["llm"]["model"])
        render_config_item("Chunks", config["chunking"]["chunk_size"])
    with col2:
        render_config_item("Temp", config["llm"]["temperature"])
        render_config_item("Top-K", config["retriever"]["k"])

    st.markdown("---")

    # ── Token Usage ──────────────────────────────────────────────────────
    st.markdown("### 📊 Token Usage")

    total_input: int = sum(
        qa.get("input_tokens", 0)
        for qa in st.session_state.get("conversation_history", [])
    )
    total_output: int = sum(
        qa.get("output_tokens", 0)
        for qa in st.session_state.get("conversation_history", [])
    )

    model_name: str = config["llm"]["model"]
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o-mini"])
    total_cost: float = (
        (total_input / 1_000_000) * pricing["input"]
        + (total_output / 1_000_000) * pricing["output"]
    )

    render_config_item("Input", f"{total_input:,}")
    render_config_item("Output", f"{total_output:,}")
    render_config_item("Cost", f"${total_cost:.6f}")

    st.markdown("---")
    st.caption("Powered by OpenAI & LangChain")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Health Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "Chat", "Manage Documents", "Blood Pressure", "Journal",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CHAT (AGENT-POWERED)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
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
                    if source == "Knowledge Base":
                        st.caption("📄 Source: Knowledge Base")
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
            logger.info("Chat history cleared by user")
            st.rerun()

    if ask_button:
        if question:
            # ── Guardrail 1: Crisis detection (regex, no LLM) ─────────────
            crisis = check_for_crisis(question)
            if crisis.detected:
                if crisis.crisis_type == CrisisType.MENTAL_HEALTH_CRISIS:
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
                for resource in crisis.resources:
                    st.markdown(resource)
            else:
                out_of_scope: bool = False
                with st.spinner("Thinking..."):
                    ask_error: str | None = None
                    try:
                        # ── Rate limit check ──────────────────────────────
                        check_and_record(
                            username,
                            config["rate_limit"]["max_requests"],
                            config["rate_limit"]["window_seconds"],
                        )

                        # ── Guardrail 2: Health topic scope (LLM check) ───
                        # Skip when a lab PDF is attached — context is clear.
                        # Pull recent context from the checkpoint so the
                        # classifier can handle follow-up queries.
                        recent_context = ""
                        if not uploaded_lab:
                            thread_id = (
                                f"{username}_{st.session_state.chat_thread_counter}"
                            )
                            state_snapshot = graph.get_state(
                                {"configurable": {"thread_id": thread_id}}
                            )
                            prev_msgs = state_snapshot.values.get(
                                "messages", []
                            )
                            # Use the last 4 messages (2 exchanges) max.
                            for msg in prev_msgs[-4:]:
                                role = msg.type  # "human" or "ai"
                                recent_context += (
                                    f"{role}: {msg.content[:200]}\n"
                                )

                        if not uploaded_lab and not is_health_related(
                            question, llm, recent_context=recent_context
                        ):
                            out_of_scope = True
                            logger.info(
                                "Query rejected by topic scope guardrail"
                            )
                        else:
                            # If a PDF is attached, extract text and include it
                            user_message: str = question
                            if uploaded_lab:
                                logger.info(
                                    "Extracting text from uploaded PDF: %s",
                                    uploaded_lab.name,
                                )
                                reader = PdfReader(
                                    BytesIO(uploaded_lab.getbuffer())
                                )
                                pdf_text = "\n".join(
                                    page.extract_text() or ""
                                    for page in reader.pages
                                )
                                if pdf_text.strip():
                                    user_message = (
                                        f"{question}\n\n"
                                        f"[Attached lab report text]\n"
                                        f"{pdf_text}"
                                    )
                                else:
                                    ask_error = (
                                        "Could not extract text from the "
                                        "uploaded PDF. The file may be "
                                        "image-based."
                                    )
                                    logger.warning(
                                        "PDF text extraction returned "
                                        "empty content"
                                    )

                            if not ask_error:
                                thread_id = (
                                    f"{username}_{st.session_state.chat_thread_counter}"
                                )
                                invoke_config = {
                                    "configurable": {"thread_id": thread_id},
                                    "recursion_limit": 10,
                                }

                                # Snapshot state before invoke so we can
                                # identify new messages and token deltas.
                                state_before = graph.get_state(invoke_config)
                                prev_values = state_before.values or {}
                                msgs_before: int = len(
                                    prev_values.get("messages", [])
                                )
                                tokens_in_before: int = prev_values.get(
                                    "input_tokens", 0
                                )
                                tokens_out_before: int = prev_values.get(
                                    "output_tokens", 0
                                )

                                result = graph.invoke(
                                    {"messages": [HumanMessage(content=user_message)]},
                                    invoke_config,
                                )
                                answer_text: str = (
                                    result["messages"][-1].content
                                )

                                # Per-turn token counts (delta).
                                turn_input_tokens: int = (
                                    result.get("input_tokens", 0)
                                    - tokens_in_before
                                )
                                turn_output_tokens: int = (
                                    result.get("output_tokens", 0)
                                    - tokens_out_before
                                )

                                # Identify tools and extract sources
                                # from only the new messages this turn.
                                new_messages = result["messages"][msgs_before:]
                                tools_used: list[str] = []
                                sources: list[str] = []
                                for msg in new_messages:
                                    if (
                                        isinstance(msg, AIMessage)
                                        and msg.tool_calls
                                    ):
                                        for tc in msg.tool_calls:
                                            if tc["name"] not in tools_used:
                                                tools_used.append(tc["name"])
                                    if (
                                        isinstance(msg, ToolMessage)
                                        and "URL:" in msg.content
                                    ):
                                        for line in msg.content.split("\n"):
                                            if line.strip().startswith("URL:"):
                                                url = (
                                                    line.strip()
                                                    .replace("URL:", "")
                                                    .strip()
                                                )
                                                if url:
                                                    sources.append(url)

                                if (
                                    not sources
                                    and "search_knowledge_base" in tools_used
                                ):
                                    sources = ["Knowledge Base"]

                                st.session_state.conversation_history.append({
                                    "question": question,
                                    "answer": answer_text,
                                    "tools_used": tools_used,
                                    "sources": sources,
                                    "input_tokens": turn_input_tokens,
                                    "output_tokens": turn_output_tokens,
                                })

                                st.session_state.question_counter += 1

                    except RateLimitExceeded as e:
                        st.warning(str(e))
                        logger.warning(
                            "Rate limit hit for user '%s'", username
                        )
                    except GraphRecursionError:
                        logger.warning("Agent hit recursion limit")
                        ask_error = (
                            "I wasn't able to complete the request. "
                            "Please try again."
                        )
                    except Exception as e:
                        logger.error("Error during agent invocation: %s", e)
                        ask_error = str(e)

                if out_of_scope:
                    st.info(
                        "This assistant is designed for health and wellness "
                        "questions only. Please ask a health-related question."
                    )
                elif ask_error:
                    st.error(f"Error: {ask_error}")
                else:
                    st.rerun()
        else:
            st.warning("Please enter a question")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MANAGE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Manage Knowledge Base")
    st.caption("Upload PDFs to expand your health knowledge base")

    st.markdown("#### Current Documents")

    if config["pdf_files"]:
        for idx, pdf_path in enumerate(config["pdf_files"]):
            col1, col2 = st.columns([5, 1])

            with col1:
                filename: str = os.path.basename(pdf_path)
                file_size: str = "Unknown size"
                if os.path.exists(pdf_path):
                    size_bytes: int = os.path.getsize(pdf_path)
                    size_kb: float = size_bytes / 1024
                    if size_kb < 1024:
                        file_size = f"{size_kb:.1f} KB"
                    else:
                        file_size = f"{size_kb / 1024:.1f} MB"

                st.markdown(f"📄 **{filename}** · {file_size}")

            with col2:
                if st.button(
                    "Delete", key=f"del_pdf_{idx}",
                    use_container_width=True,
                ):
                    try:
                        logger.info("Deleting document: %s", pdf_path)
                        remove_pdf_from_vectorstore(vectorstore, pdf_path)
                        st.session_state.pdf_files.remove(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                        st.cache_resource.clear()
                        logger.info("Document deleted successfully")
                    except Exception as e:
                        logger.error("Error deleting document: %s", e)
                        st.error(f"Error deleting file: {e}")

                    st.rerun()
    else:
        st.info("No documents in knowledge base yet")

    st.markdown("---")

    st.markdown("#### Add New Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="kb_pdf_upload",
        help=(
            "Upload health-related PDF documents to add "
            "to your knowledge base"
        ),
    )

    if uploaded_pdf:
        st.success(f"✓ Selected: **{uploaded_pdf.name}**")

        if st.button(
            "Add to Knowledge Base", type="primary",
            use_container_width=True,
        ):
            add_success: bool = False
            with st.spinner("Adding document to knowledge base..."):
                try:
                    pdf_dir: str = "data"
                    os.makedirs(pdf_dir, exist_ok=True)

                    pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)
                    logger.info("Saving uploaded PDF to %s", pdf_path)

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    if pdf_path not in st.session_state.pdf_files:
                        st.session_state.pdf_files.append(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                    add_pdf_to_vectorstore(
                        vectorstore,
                        pdf_path,
                        chunk_size=config["chunking"]["chunk_size"],
                        chunk_overlap=config["chunking"]["chunk_overlap"],
                    )

                    st.cache_resource.clear()
                    add_success = True
                    logger.info("Document added successfully: %s", pdf_path)
                except Exception as e:
                    logger.error("Error adding document: %s", e)
                    st.error(f"❌ Error adding document: {e}")

            if add_success:
                st.rerun()
    else:
        st.info("👆 Choose a PDF file to add to your knowledge base")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: BLOOD PRESSURE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Blood Pressure Tracker")
    st.caption("Record and monitor your blood pressure over time")

    # ── Entry Form ───────────────────────────────────────────────────────
    with st.form("bp_form", clear_on_submit=True):
        bp_cols = st.columns([1, 1, 1, 1, 1])

        with bp_cols[0]:
            bp_date = st.date_input("Date", key="bp_date")
        with bp_cols[1]:
            bp_time = st.time_input("Time", key="bp_time")
        with bp_cols[2]:
            bp_systolic = st.number_input(
                "Systolic", min_value=60, max_value=250, value=120,
                key="bp_systolic",
            )
        with bp_cols[3]:
            bp_diastolic = st.number_input(
                "Diastolic", min_value=30, max_value=150, value=80,
                key="bp_diastolic",
            )
        with bp_cols[4]:
            bp_pulse = st.number_input(
                "Pulse", min_value=30, max_value=220, value=72,
                key="bp_pulse",
            )

        if st.form_submit_button(
            "Save Reading", use_container_width=True,
        ):
            reading: dict[str, Any] = {
                "date": bp_date.strftime("%Y-%m-%d"),
                "time": bp_time.strftime("%H:%M"),
                "systolic": bp_systolic,
                "diastolic": bp_diastolic,
                "pulse": bp_pulse,
            }
            st.session_state.bp_readings.append(reading)
            save_user_data(username)
            logger.info("Blood pressure reading saved: %s", reading)
            st.rerun()

    # ── Chart ────────────────────────────────────────────────────────────
    if st.session_state.bp_readings:
        st.markdown("---")
        st.markdown("#### Trends")

        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame(st.session_state.bp_readings)
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.sort_values("datetime")
        df["label"] = df["datetime"].dt.strftime("%b %d, %Y")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Systolic",
            x=df["label"],
            y=df["systolic"],
            marker_color="#EF4444",
        ))
        fig.add_trace(go.Bar(
            name="Diastolic",
            x=df["label"],
            y=df["diastolic"],
            marker_color="#3B82F6",
        ))
        fig.add_trace(go.Bar(
            name="Pulse",
            x=df["label"],
            y=df.get("pulse", pd.Series([0] * len(df))),
            marker_color="#10B981",
        ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Date",
            yaxis_title="mmHg / BPM",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Readings ──────────────────────────────────────────────────────
        st.markdown("#### Readings")

        for idx, reading in enumerate(
            reversed(st.session_state.bp_readings)
        ):
            real_idx: int = (
                len(st.session_state.bp_readings) - 1 - idx
            )

            with st.container(border=True):
                col_date, col_sys, col_dia, col_pulse, col_del = st.columns(
                    [2, 1.5, 1.5, 1.5, 1]
                )

                with col_date:
                    parsed_date = datetime.strptime(
                        reading["date"], "%Y-%m-%d"
                    )
                    display_date = (
                        f"{parsed_date.strftime('%B')} "
                        f"{parsed_date.day}, "
                        f"{parsed_date.year}"
                    )
                    st.markdown(
                        f"**{display_date}**  \n"
                        f"{reading.get('time', '')}"
                    )
                with col_sys:
                    st.metric(
                        label="Systolic",
                        value=f"{reading['systolic']}",
                    )
                with col_dia:
                    st.metric(
                        label="Diastolic",
                        value=f"{reading['diastolic']}",
                    )
                with col_pulse:
                    st.metric(
                        label="Pulse",
                        value=f"{reading.get('pulse', '—')}",
                    )
                with col_del:
                    st.markdown("")
                    if st.button(
                        "Delete",
                        key=f"del_bp_{real_idx}",
                        use_container_width=True,
                    ):
                        st.session_state.bp_readings.pop(real_idx)
                        save_user_data(username)
                        logger.info("Blood pressure reading deleted")
                        st.rerun()
    else:
        st.info(
            "No readings yet. Start tracking your blood pressure!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: HEALTH JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")

    col1, col2 = st.columns([3, 1])

    with col1:
        journal_title: str = st.text_input(
            "Title",
            placeholder="Entry title...",
            key=f"journal_title_{st.session_state.journal_form_key}",
        )

        journal_entry: str = st.text_area(
            "Entry",
            placeholder="How are you feeling today?",
            height=120,
            key=f"journal_entry_{st.session_state.journal_form_key}",
        )

        uploaded_file = st.file_uploader(
            "Attachment (optional)",
            type=["pdf", "png", "jpg", "jpeg", "gif"],
            key=f"journal_file_{st.session_state.file_uploader_key}",
        )

    with col2:
        journal_date = st.date_input("Date", key="journal_date")

        st.markdown("")

        if st.button("Save Entry", use_container_width=True):
            if journal_title and journal_entry:
                entry_data: dict[str, Any] = {
                    "title": journal_title,
                    "date": journal_date.strftime("%Y-%m-%d"),
                    "entry": journal_entry,
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }

                if uploaded_file is not None:
                    attachment_dir: str = (
                        f"journal_attachments/{username}"
                    )
                    os.makedirs(attachment_dir, exist_ok=True)

                    timestamp: str = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    file_extension: str = (
                        uploaded_file.name.rsplit(".", 1)[-1].lower()
                    )
                    safe_filename: str = (
                        f"{timestamp}_{uploaded_file.name}"
                    )
                    file_path: str = os.path.join(
                        attachment_dir, safe_filename
                    )

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    entry_data["attachment"] = {
                        "filename": uploaded_file.name,
                        "filepath": file_path,
                        "type": file_extension,
                    }
                    logger.info(
                        "Journal attachment saved: %s", file_path,
                    )

                st.session_state.journal_entries.append(entry_data)
                save_user_data(username)
                st.session_state.file_uploader_key += 1
                st.session_state.journal_form_key += 1
                logger.info("Journal entry saved: %s", journal_title)
                st.rerun()
            else:
                st.warning("Please enter both title and entry")

    st.markdown("---")

    # Past Entries
    if st.session_state.journal_entries:
        st.markdown("#### Past Entries")

        for entry in reversed(st.session_state.journal_entries):
            attachment_icon: str = " 📎" if "attachment" in entry else ""
            is_editing: bool = (
                st.session_state.editing_entry == entry["timestamp"]
            )

            with st.container(border=True):
                if is_editing:
                    st.markdown("**✏️ Editing Entry**")

                    edited_title: str = st.text_input(
                        "Title",
                        value=entry.get("title", ""),
                        key=f"edit_title_{entry['timestamp']}",
                    )

                    edited_entry_text: str = st.text_area(
                        "Entry",
                        value=entry["entry"],
                        height=120,
                        key=f"edit_entry_{entry['timestamp']}",
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button(
                            "Save Changes",
                            key=f"save_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e["timestamp"] == entry["timestamp"]:
                                    st.session_state.journal_entries[idx][
                                        "title"
                                    ] = edited_title
                                    st.session_state.journal_entries[idx][
                                        "entry"
                                    ] = edited_entry_text
                                    break
                            save_user_data(username)
                            st.session_state.editing_entry = None
                            logger.info("Journal entry updated")
                            st.success("Entry updated!")
                            st.rerun()

                    with col_cancel:
                        if st.button(
                            "Cancel",
                            key=f"cancel_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_entry = None
                            st.rerun()
                else:
                    parsed_journal_date = datetime.strptime(
                        entry["date"], "%Y-%m-%d"
                    )
                    journal_display_date = (
                        f"{parsed_journal_date.strftime('%B')} "
                        f"{parsed_journal_date.day}, "
                        f"{parsed_journal_date.year}"
                    )
                    st.markdown(
                        f"**{journal_display_date} — "
                        f"{entry.get('title', 'Untitled')}"
                        f"{attachment_icon}**"
                    )
                    st.markdown(entry["entry"])

                    if "attachment" in entry:
                        st.markdown("---")
                        attachment: dict[str, str] = entry["attachment"]
                        file_type: str = attachment["type"]

                        if file_type in ["png", "jpg", "jpeg", "gif"]:
                            st.image(
                                attachment["filepath"],
                                caption=attachment["filename"],
                                use_container_width=True,
                            )
                        elif file_type == "pdf":
                            st.caption(f"📄 {attachment['filename']}")
                            with open(attachment["filepath"], "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name=attachment["filename"],
                                    mime="application/pdf",
                                    key=f"dl_{entry['timestamp']}",
                                )

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit Entry",
                            key=f"edit_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_entry = (
                                entry["timestamp"]
                            )
                            st.rerun()

                    with col_delete:
                        if st.button(
                            "Delete Entry",
                            key=f"del_{entry['timestamp']}",
                            use_container_width=True,
                        ):
                            if (
                                "attachment" in entry
                                and "filepath" in entry["attachment"]
                            ):
                                filepath: str = os.path.normpath(
                                    entry["attachment"]["filepath"]
                                )
                                if os.path.exists(filepath):
                                    try:
                                        os.remove(filepath)
                                    except OSError as e:
                                        logger.error(
                                            "Failed to delete attachment "
                                            "%s: %s", filepath, e,
                                        )

                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e["timestamp"] == entry["timestamp"]:
                                    st.session_state.journal_entries.pop(idx)
                                    break

                            save_user_data(username)
                            logger.info("Journal entry deleted")
                            st.rerun()
    else:
        st.info(
            "No journal entries yet. Start tracking your health journey!"
        )
