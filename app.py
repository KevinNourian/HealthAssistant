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
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import load_config, save_config, validate_api_keys, SOURCE_KNOWLEDGE_BASE
from core.auth import load_authenticator, render_login
from core.user_data import init_session_state, save_user_data
from core.tools import create_tools
from core.agent import build_graph
from core.guardrails import CrisisType
from core.utils import process_query, calculate_session_cost
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
graph = build_graph(
    llm,
    tools,
    checkpointer=checkpointer,
    max_context_messages=config["llm"]["max_context_messages"],
)


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

    cost = calculate_session_cost(
        st.session_state.get("conversation_history", []),
        config["llm"]["model"],
    )

    render_config_item("Input", f"{cost.total_input:,}")
    render_config_item("Output", f"{cost.total_output:,}")
    render_config_item("Cost", f"${cost.total_cost:.6f}")

    st.markdown("---")
    st.caption("Powered by OpenAI & LangChain")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Health Assistant")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Chat", "Manage Documents", "Blood Pressure", "Medical Visit",
    "Vaccination", "Prescription", "Lab Test", "Journal",
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
            with st.spinner("Thinking..."):
                result = process_query(
                    graph=graph,
                    llm=llm,
                    question=question,
                    uploaded_lab=uploaded_lab,
                    username=username,
                    chat_thread_counter=st.session_state.chat_thread_counter,
                    config=config,
                )

            # ── Display result ────────────────────────────────────────────
            if result.crisis:
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
# TAB 4: MEDICAL VISIT
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Medical Visit")
    st.caption("Record and track your medical appointments")

    # ── Entry Form ───────────────────────────────────────────────────────
    mv_key = st.session_state.mv_form_key
    mv_cols = st.columns([1, 1, 2])

    with mv_cols[0]:
        mv_date = st.date_input("Date", key=f"mv_date_{mv_key}")
    with mv_cols[1]:
        mv_time = st.time_input("Time", key=f"mv_time_{mv_key}")
    with mv_cols[2]:
        mv_doctor = st.text_input("Doctor", key=f"mv_doctor_{mv_key}")

    mv_notes = st.text_area(
        "Notes (Markdown supported)",
        key=f"mv_notes_{mv_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Visit", use_container_width=True):
            if not mv_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                visit: dict[str, Any] = {
                    "date": mv_date.strftime("%Y-%m-%d"),
                    "time": mv_time.strftime("%H:%M"),
                    "doctor": mv_doctor.strip(),
                    "notes": mv_notes.strip(),
                }
                st.session_state.medical_visits.append(visit)
                save_user_data(username)
                st.session_state.mv_form_key += 1
                logger.info("Medical visit saved: %s", visit)
                st.rerun()
    with col_cancel:
        if st.button("Cancel", key="mv_cancel", use_container_width=True):
            st.session_state.mv_form_key += 1
            st.rerun()

    # ── Visits ────────────────────────────────────────────────────────────
    if st.session_state.medical_visits:
        st.markdown("---")
        st.markdown("#### Visits")

        for idx, visit in enumerate(
            reversed(st.session_state.medical_visits)
        ):
            real_idx: int = (
                len(st.session_state.medical_visits) - 1 - idx
            )
            is_editing_visit: bool = (
                st.session_state.editing_visit == real_idx
            )

            with st.container(border=True):
                if is_editing_visit:
                    st.markdown("**Editing Visit**")

                    edit_mv_cols = st.columns([1, 1, 2])
                    with edit_mv_cols[0]:
                        edited_mv_date = st.date_input(
                            "Date",
                            value=datetime.strptime(
                                visit["date"], "%Y-%m-%d"
                            ).date(),
                            key=f"edit_mv_date_{real_idx}",
                        )
                    with edit_mv_cols[1]:
                        edited_mv_time = st.time_input(
                            "Time",
                            value=datetime.strptime(
                                visit["time"], "%H:%M"
                            ).time(),
                            key=f"edit_mv_time_{real_idx}",
                        )
                    with edit_mv_cols[2]:
                        edited_mv_doctor = st.text_input(
                            "Doctor",
                            value=visit["doctor"],
                            key=f"edit_mv_doctor_{real_idx}",
                        )

                    edited_mv_notes = st.text_area(
                        "Notes (Markdown supported)",
                        value=visit.get("notes", ""),
                        height=120,
                        key=f"edit_mv_notes_{real_idx}",
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button(
                            "Save Changes",
                            key=f"save_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.medical_visits[real_idx] = {
                                "date": edited_mv_date.strftime("%Y-%m-%d"),
                                "time": edited_mv_time.strftime("%H:%M"),
                                "doctor": edited_mv_doctor.strip(),
                                "notes": edited_mv_notes.strip(),
                            }
                            save_user_data(username)
                            st.session_state.editing_visit = None
                            logger.info("Medical visit updated")
                            st.rerun()
                    with col_cancel:
                        if st.button(
                            "Cancel",
                            key=f"cancel_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_visit = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_del = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            visit["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(
                            f"**{display_date}**  \n"
                            f"{visit.get('time', '')}"
                        )
                    with col_doctor:
                        st.markdown(f"**Dr.** {visit['doctor']}")
                    with col_del:
                        st.markdown("")

                    if visit.get("notes"):
                        st.markdown("---")
                        st.markdown(visit["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_visit = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_mv_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.medical_visits.pop(real_idx)
                            save_user_data(username)
                            logger.info("Medical visit deleted")
                            st.rerun()
    else:
        st.info(
            "No visits yet. Start tracking your medical appointments!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: VACCINATION
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Vaccination")
    st.caption("Record and track your vaccinations")

    # ── Entry Form ───────────────────────────────────────────────────────
    vax_key = st.session_state.vax_form_key
    vax_cols = st.columns([1, 2])

    with vax_cols[0]:
        vax_date = st.date_input("Date", key=f"vax_date_{vax_key}")
    with vax_cols[1]:
        vax_name = st.text_input(
            "Name of Vaccination", key=f"vax_name_{vax_key}",
        )

    vax_notes = st.text_area(
        "Notes (Markdown supported)",
        key=f"vax_notes_{vax_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Vaccination", use_container_width=True):
            if not vax_name.strip():
                st.warning("Please enter the name of the vaccination.")
            else:
                vaccination: dict[str, Any] = {
                    "date": vax_date.strftime("%Y-%m-%d"),
                    "name": vax_name.strip(),
                    "notes": vax_notes.strip(),
                }
                st.session_state.vaccinations.append(vaccination)
                save_user_data(username)
                st.session_state.vax_form_key += 1
                logger.info("Vaccination saved: %s", vaccination)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="vax_cancel", use_container_width=True,
        ):
            st.session_state.vax_form_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.vaccinations:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, vax in enumerate(
            reversed(st.session_state.vaccinations)
        ):
            real_idx: int = (
                len(st.session_state.vaccinations) - 1 - idx
            )
            is_editing_vax: bool = (
                st.session_state.editing_vaccination == real_idx
            )

            with st.container(border=True):
                if is_editing_vax:
                    st.markdown("**Editing Vaccination**")

                    edit_vax_cols = st.columns([1, 2])
                    with edit_vax_cols[0]:
                        edited_vax_date = st.date_input(
                            "Date",
                            value=datetime.strptime(
                                vax["date"], "%Y-%m-%d"
                            ).date(),
                            key=f"edit_vax_date_{real_idx}",
                        )
                    with edit_vax_cols[1]:
                        edited_vax_name = st.text_input(
                            "Name of Vaccination",
                            value=vax["name"],
                            key=f"edit_vax_name_{real_idx}",
                        )

                    edited_vax_notes = st.text_area(
                        "Notes (Markdown supported)",
                        value=vax.get("notes", ""),
                        height=120,
                        key=f"edit_vax_notes_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.vaccinations[real_idx] = {
                                "date": edited_vax_date.strftime("%Y-%m-%d"),
                                "name": edited_vax_name.strip(),
                                "notes": edited_vax_notes.strip(),
                            }
                            save_user_data(username)
                            st.session_state.editing_vaccination = None
                            logger.info("Vaccination updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_vaccination = None
                            st.rerun()
                else:
                    col_date, col_name, col_del = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            vax["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_name:
                        st.markdown(f"**{vax['name']}**")
                    with col_del:
                        st.markdown("")

                    if vax.get("notes"):
                        st.markdown("---")
                        st.markdown(vax["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_vaccination = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_vax_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.vaccinations.pop(real_idx)
                            save_user_data(username)
                            logger.info("Vaccination deleted")
                            st.rerun()
    else:
        st.info(
            "No vaccinations yet. Start tracking your vaccinations!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: PRESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### Prescription")
    st.caption("Record and track your prescriptions")

    # ── Entry Form ───────────────────────────────────────────────────────
    rx_key = st.session_state.rx_form_key
    drug_key = st.session_state.rx_drug_key

    rx_date = st.date_input("Date", key=f"rx_date_{rx_key}")

    # ── Drug / Dose builder ──────────────────────────────────────────────
    st.markdown("**Drugs & Doses**")

    # Display already-added drugs
    for d_idx, drug_entry in enumerate(
        st.session_state.pending_rx_drugs
    ):
        is_editing_drug: bool = (
            st.session_state.editing_rx_drug
            == f"pending_{d_idx}"
        )
        with st.container(border=True):
            if is_editing_drug:
                ed_cols = st.columns([2, 2, 1, 1])
                with ed_cols[0]:
                    edited_drug = st.text_input(
                        "Drug",
                        value=drug_entry["drug"],
                        key=f"pedit_drug_{d_idx}_{rx_key}",
                        label_visibility="collapsed",
                    )
                with ed_cols[1]:
                    edited_dose = st.text_input(
                        "Dose",
                        value=drug_entry["dose"],
                        key=f"pedit_dose_{d_idx}_{rx_key}",
                        label_visibility="collapsed",
                    )
                with ed_cols[2]:
                    if st.button(
                        "Save",
                        key=f"pedit_save_{d_idx}_{rx_key}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_rx_drugs[d_idx] = {
                            "drug": edited_drug.strip(),
                            "dose": edited_dose.strip(),
                        }
                        st.session_state.editing_rx_drug = None
                        st.rerun()
                with ed_cols[3]:
                    if st.button(
                        "Cancel",
                        key=f"pedit_cancel_{d_idx}_{rx_key}",
                        use_container_width=True,
                    ):
                        st.session_state.editing_rx_drug = None
                        st.rerun()
            else:
                d_col_drug, d_col_sep, d_col_dose, d_col_edit, d_col_trash = (
                    st.columns([2, 0.15, 2, 0.5, 0.5])
                )
                with d_col_drug:
                    st.markdown(f"**{drug_entry['drug']}**")
                with d_col_sep:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with d_col_dose:
                    st.markdown(drug_entry["dose"])
                with d_col_edit:
                    if st.button(
                        "\u270F\uFE0F",
                        key=f"pen_rx_{d_idx}_{rx_key}",
                    ):
                        st.session_state.editing_rx_drug = (
                            f"pending_{d_idx}"
                        )
                        st.rerun()
                with d_col_trash:
                    if st.button(
                        "\U0001f5d1\uFE0F",
                        key=f"trash_rx_{d_idx}_{rx_key}",
                    ):
                        st.session_state.pending_rx_drugs.pop(d_idx)
                        st.rerun()

    # Input row for new drug/dose
    add_cols = st.columns([2, 2, 1])
    with add_cols[0]:
        new_drug = st.text_input(
            "Drug", key=f"rx_drug_{drug_key}", label_visibility="collapsed",
            placeholder="Drug name",
        )
    with add_cols[1]:
        new_dose = st.text_input(
            "Dose", key=f"rx_dose_{drug_key}", label_visibility="collapsed",
            placeholder="Dose",
        )
    with add_cols[2]:
        if st.button("Add", key=f"rx_add_{drug_key}",
                      use_container_width=True):
            if new_drug.strip() and new_dose.strip():
                st.session_state.pending_rx_drugs.append({
                    "drug": new_drug.strip(),
                    "dose": new_dose.strip(),
                })
                st.session_state.rx_drug_key += 1
                st.rerun()
            else:
                st.warning("Please enter both drug and dose.")

    rx_doctor = st.text_input("Doctor", key=f"rx_doctor_{rx_key}")

    rx_notes = st.text_area(
        "Notes (Markdown supported)",
        key=f"rx_notes_{rx_key}",
        height=120,
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Prescription", use_container_width=True):
            if not st.session_state.pending_rx_drugs:
                st.warning("Please add at least one drug and dose.")
            elif not rx_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                prescription: dict[str, Any] = {
                    "date": rx_date.strftime("%Y-%m-%d"),
                    "drugs": list(st.session_state.pending_rx_drugs),
                    "doctor": rx_doctor.strip(),
                    "notes": rx_notes.strip(),
                }
                st.session_state.prescriptions.append(prescription)
                st.session_state.pending_rx_drugs = []
                save_user_data(username)
                st.session_state.rx_form_key += 1
                st.session_state.rx_drug_key += 1
                logger.info("Prescription saved: %s", prescription)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="rx_cancel", use_container_width=True,
        ):
            st.session_state.pending_rx_drugs = []
            st.session_state.rx_form_key += 1
            st.session_state.rx_drug_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.prescriptions:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, rx in enumerate(
            reversed(st.session_state.prescriptions)
        ):
            real_idx: int = (
                len(st.session_state.prescriptions) - 1 - idx
            )
            is_editing_rx: bool = (
                st.session_state.editing_prescription == real_idx
            )

            with st.container(border=True):
                if is_editing_rx:
                    st.markdown("**Editing Prescription**")

                    edited_rx_date = st.date_input(
                        "Date",
                        value=datetime.strptime(
                            rx["date"], "%Y-%m-%d"
                        ).date(),
                        key=f"edit_rx_date_{real_idx}",
                    )

                    # Editable drug list stored in session state
                    edit_drugs_key = f"edit_rx_drugs_{real_idx}"
                    if edit_drugs_key not in st.session_state:
                        st.session_state[edit_drugs_key] = list(
                            rx.get("drugs", [])
                        )

                    st.markdown("**Drugs & Doses**")
                    for d_idx, drug_entry in enumerate(
                        st.session_state[edit_drugs_key]
                    ):
                        is_editing_ed: bool = (
                            st.session_state.editing_rx_drug
                            == f"erx_{real_idx}_{d_idx}"
                        )
                        with st.container(border=True):
                            if is_editing_ed:
                                ed_cols = st.columns([2, 2, 1, 1])
                                with ed_cols[0]:
                                    eed_drug = st.text_input(
                                        "Drug",
                                        value=drug_entry["drug"],
                                        key=f"eed_drug_{real_idx}_{d_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ed_cols[1]:
                                    eed_dose = st.text_input(
                                        "Dose",
                                        value=drug_entry["dose"],
                                        key=f"eed_dose_{real_idx}_{d_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ed_cols[2]:
                                    if st.button(
                                        "Save",
                                        key=f"eed_save_{real_idx}_{d_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[
                                            edit_drugs_key
                                        ][d_idx] = {
                                            "drug": eed_drug.strip(),
                                            "dose": eed_dose.strip(),
                                        }
                                        st.session_state.editing_rx_drug = (
                                            None
                                        )
                                        st.rerun()
                                with ed_cols[3]:
                                    if st.button(
                                        "Cancel",
                                        key=f"eed_cancel_{real_idx}_{d_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.editing_rx_drug = (
                                            None
                                        )
                                        st.rerun()
                            else:
                                d_col_drug, d_col_sep, d_col_dose, d_col_edit, d_col_trash = (
                                    st.columns([2, 0.15, 2, 0.5, 0.5])
                                )
                                with d_col_drug:
                                    st.markdown(
                                        f"**{drug_entry['drug']}**"
                                    )
                                with d_col_sep:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with d_col_dose:
                                    st.markdown(drug_entry["dose"])
                                with d_col_edit:
                                    if st.button(
                                        "\u270F\uFE0F",
                                        key=f"epen_rx_{real_idx}_{d_idx}",
                                    ):
                                        st.session_state.editing_rx_drug = (
                                            f"erx_{real_idx}_{d_idx}"
                                        )
                                        st.rerun()
                                with d_col_trash:
                                    if st.button(
                                        "\U0001f5d1\uFE0F",
                                        key=f"etrash_rx_{real_idx}_{d_idx}",
                                    ):
                                        st.session_state[
                                            edit_drugs_key
                                        ].pop(d_idx)
                                        st.rerun()

                    edit_add_cols = st.columns([2, 2, 1])
                    with edit_add_cols[0]:
                        edit_new_drug = st.text_input(
                            "Drug",
                            key=f"edit_rx_newdrug_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Drug name",
                        )
                    with edit_add_cols[1]:
                        edit_new_dose = st.text_input(
                            "Dose",
                            key=f"edit_rx_newdose_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Dose",
                        )
                    with edit_add_cols[2]:
                        if st.button(
                            "Add",
                            key=f"edit_rx_add_{real_idx}",
                            use_container_width=True,
                        ):
                            if (
                                edit_new_drug.strip()
                                and edit_new_dose.strip()
                            ):
                                st.session_state[edit_drugs_key].append({
                                    "drug": edit_new_drug.strip(),
                                    "dose": edit_new_dose.strip(),
                                })
                                st.rerun()

                    edited_rx_doctor = st.text_input(
                        "Doctor",
                        value=rx["doctor"],
                        key=f"edit_rx_doctor_{real_idx}",
                    )

                    edited_rx_notes = st.text_area(
                        "Notes (Markdown supported)",
                        value=rx.get("notes", ""),
                        height=120,
                        key=f"edit_rx_notes_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.prescriptions[real_idx] = {
                                "date": edited_rx_date.strftime("%Y-%m-%d"),
                                "drugs": list(
                                    st.session_state[edit_drugs_key]
                                ),
                                "doctor": edited_rx_doctor.strip(),
                                "notes": edited_rx_notes.strip(),
                            }
                            del st.session_state[edit_drugs_key]
                            save_user_data(username)
                            st.session_state.editing_prescription = None
                            logger.info("Prescription updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            if edit_drugs_key in st.session_state:
                                del st.session_state[edit_drugs_key]
                            st.session_state.editing_prescription = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_spacer = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            rx["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_doctor:
                        st.markdown(f"**Dr.** {rx['doctor']}")

                    # Drug list
                    for drug_entry in rx.get("drugs", []):
                        with st.container(border=True):
                            d_col_drug, d_col_sep, d_col_dose = st.columns(
                                [2, 0.15, 2]
                            )
                            with d_col_drug:
                                st.markdown(
                                    f"**{drug_entry['drug']}**"
                                )
                            with d_col_sep:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with d_col_dose:
                                st.markdown(drug_entry["dose"])

                    if rx.get("notes"):
                        st.markdown("---")
                        st.markdown(rx["notes"])

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_prescription = (
                                real_idx
                            )
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_rx_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.prescriptions.pop(real_idx)
                            save_user_data(username)
                            logger.info("Prescription deleted")
                            st.rerun()
    else:
        st.info(
            "No prescriptions yet. Start tracking your prescriptions!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: LAB TEST
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("### Lab Test")
    st.caption("Record and track your lab test results")

    # ── Entry Form ───────────────────────────────────────────────────────
    lab_key = st.session_state.lab_form_key
    item_key = st.session_state.lab_item_key

    lab_date = st.date_input("Date", key=f"lab_date_{lab_key}")

    # ── Test / Result / Normal builder ───────────────────────────────────
    st.markdown("**Type**")

    # Display already-added test items
    for t_idx, test_item in enumerate(
        st.session_state.pending_lab_items
    ):
        is_editing_item: bool = (
            st.session_state.editing_lab_item
            == f"pending_{t_idx}"
        )
        with st.container(border=True):
            if is_editing_item:
                ei_cols = st.columns([2, 2, 2, 1, 1])
                with ei_cols[0]:
                    edited_test = st.text_input(
                        "Test",
                        value=test_item["test"],
                        key=f"ledit_test_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[1]:
                    edited_result = st.text_input(
                        "Result",
                        value=test_item["result"],
                        key=f"ledit_result_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[2]:
                    edited_normal = st.text_input(
                        "Normal",
                        value=test_item["normal"],
                        key=f"ledit_normal_{t_idx}_{lab_key}",
                        label_visibility="collapsed",
                    )
                with ei_cols[3]:
                    if st.button(
                        "Save",
                        key=f"ledit_save_{t_idx}_{lab_key}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_lab_items[t_idx] = {
                            "test": edited_test.strip(),
                            "result": edited_result.strip(),
                            "normal": edited_normal.strip(),
                        }
                        st.session_state.editing_lab_item = None
                        st.rerun()
                with ei_cols[4]:
                    if st.button(
                        "Cancel",
                        key=f"ledit_cancel_{t_idx}_{lab_key}",
                        use_container_width=True,
                    ):
                        st.session_state.editing_lab_item = None
                        st.rerun()
            else:
                t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal, t_col_edit, t_col_trash = (
                    st.columns([2, 0.15, 2, 0.15, 2, 0.5, 0.5])
                )
                with t_col_test:
                    st.markdown(f"**{test_item['test']}**")
                with t_col_sep1:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with t_col_result:
                    st.markdown(test_item["result"])
                with t_col_sep2:
                    st.markdown(
                        "<div style='border-left: 2px solid #ccc;"
                        " height: 100%; min-height: 30px;'></div>",
                        unsafe_allow_html=True,
                    )
                with t_col_normal:
                    st.markdown(test_item["normal"])
                with t_col_edit:
                    if st.button(
                        "\u270F\uFE0F",
                        key=f"pen_lab_{t_idx}_{lab_key}",
                    ):
                        st.session_state.editing_lab_item = (
                            f"pending_{t_idx}"
                        )
                        st.rerun()
                with t_col_trash:
                    if st.button(
                        "\U0001f5d1\uFE0F",
                        key=f"trash_lab_{t_idx}_{lab_key}",
                    ):
                        st.session_state.pending_lab_items.pop(t_idx)
                        st.rerun()

    # Input row for new test item
    add_cols = st.columns([2, 2, 2, 1])
    with add_cols[0]:
        new_test = st.text_input(
            "Test", key=f"lab_test_{item_key}",
            label_visibility="collapsed", placeholder="Test",
        )
    with add_cols[1]:
        new_result = st.text_input(
            "Result", key=f"lab_result_{item_key}",
            label_visibility="collapsed", placeholder="Result",
        )
    with add_cols[2]:
        new_normal = st.text_input(
            "Normal", key=f"lab_normal_{item_key}",
            label_visibility="collapsed", placeholder="Normal",
        )
    with add_cols[3]:
        if st.button("Add", key=f"lab_add_{item_key}",
                      use_container_width=True):
            if (
                new_test.strip()
                and new_result.strip()
                and new_normal.strip()
            ):
                st.session_state.pending_lab_items.append({
                    "test": new_test.strip(),
                    "result": new_result.strip(),
                    "normal": new_normal.strip(),
                })
                st.session_state.lab_item_key += 1
                st.rerun()
            else:
                st.warning("Please enter test, result, and normal range.")

    lab_doctor = st.text_input("Doctor", key=f"lab_doctor_{lab_key}")

    lab_notes = st.text_area(
        "Notes (Markdown supported)",
        key=f"lab_notes_{lab_key}",
        height=120,
    )

    lab_uploaded_file = st.file_uploader(
        "Attach PDF (optional)",
        type=["pdf"],
        key=f"lab_file_{st.session_state.lab_file_key}",
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save Lab Test", use_container_width=True):
            if not st.session_state.pending_lab_items:
                st.warning("Please add at least one test result.")
            elif not lab_doctor.strip():
                st.warning("Please enter a doctor name.")
            else:
                lab_test: dict[str, Any] = {
                    "date": lab_date.strftime("%Y-%m-%d"),
                    "tests": list(st.session_state.pending_lab_items),
                    "doctor": lab_doctor.strip(),
                    "notes": lab_notes.strip(),
                }

                if lab_uploaded_file is not None:
                    lab_attach_dir: str = (
                        f"lab_attachments/{username}"
                    )
                    os.makedirs(lab_attach_dir, exist_ok=True)
                    lab_timestamp: str = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    lab_safe_name: str = (
                        f"{lab_timestamp}_{lab_uploaded_file.name}"
                    )
                    lab_file_path: str = os.path.join(
                        lab_attach_dir, lab_safe_name
                    )
                    with open(lab_file_path, "wb") as f:
                        f.write(lab_uploaded_file.getbuffer())
                    lab_test["attachment"] = {
                        "filename": lab_uploaded_file.name,
                        "filepath": lab_file_path,
                    }
                    logger.info(
                        "Lab attachment saved: %s", lab_file_path,
                    )

                st.session_state.lab_tests.append(lab_test)
                st.session_state.pending_lab_items = []
                save_user_data(username)
                st.session_state.lab_form_key += 1
                st.session_state.lab_item_key += 1
                st.session_state.lab_file_key += 1
                logger.info("Lab test saved: %s", lab_test)
                st.rerun()
    with col_cancel:
        if st.button(
            "Cancel", key="lab_cancel", use_container_width=True,
        ):
            st.session_state.pending_lab_items = []
            st.session_state.lab_form_key += 1
            st.session_state.lab_item_key += 1
            st.session_state.lab_file_key += 1
            st.rerun()

    # ── Records ──────────────────────────────────────────────────────────
    if st.session_state.lab_tests:
        st.markdown("---")
        st.markdown("#### Records")

        for idx, lt in enumerate(
            reversed(st.session_state.lab_tests)
        ):
            real_idx: int = (
                len(st.session_state.lab_tests) - 1 - idx
            )
            is_editing_lt: bool = (
                st.session_state.editing_lab_test == real_idx
            )

            with st.container(border=True):
                if is_editing_lt:
                    st.markdown("**Editing Lab Test**")

                    edited_lab_date = st.date_input(
                        "Date",
                        value=datetime.strptime(
                            lt["date"], "%Y-%m-%d"
                        ).date(),
                        key=f"edit_lab_date_{real_idx}",
                    )

                    # Editable test list stored in session state
                    edit_tests_key = f"edit_lab_tests_{real_idx}"
                    if edit_tests_key not in st.session_state:
                        st.session_state[edit_tests_key] = list(
                            lt.get("tests", [])
                        )

                    st.markdown("**Type**")
                    for t_idx, test_item in enumerate(
                        st.session_state[edit_tests_key]
                    ):
                        is_editing_ei: bool = (
                            st.session_state.editing_lab_item
                            == f"elt_{real_idx}_{t_idx}"
                        )
                        with st.container(border=True):
                            if is_editing_ei:
                                ei_cols = st.columns([2, 2, 2, 1, 1])
                                with ei_cols[0]:
                                    eet_test = st.text_input(
                                        "Test",
                                        value=test_item["test"],
                                        key=f"eet_test_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[1]:
                                    eet_result = st.text_input(
                                        "Result",
                                        value=test_item["result"],
                                        key=f"eet_result_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[2]:
                                    eet_normal = st.text_input(
                                        "Normal",
                                        value=test_item["normal"],
                                        key=f"eet_normal_{real_idx}_{t_idx}",
                                        label_visibility="collapsed",
                                    )
                                with ei_cols[3]:
                                    if st.button(
                                        "Save",
                                        key=f"eet_save_{real_idx}_{t_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[
                                            edit_tests_key
                                        ][t_idx] = {
                                            "test": eet_test.strip(),
                                            "result": eet_result.strip(),
                                            "normal": eet_normal.strip(),
                                        }
                                        st.session_state.editing_lab_item = (
                                            None
                                        )
                                        st.rerun()
                                with ei_cols[4]:
                                    if st.button(
                                        "Cancel",
                                        key=f"eet_cancel_{real_idx}_{t_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.editing_lab_item = (
                                            None
                                        )
                                        st.rerun()
                            else:
                                t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal, t_col_edit, t_col_trash = (
                                    st.columns([2, 0.15, 2, 0.15, 2, 0.5, 0.5])
                                )
                                with t_col_test:
                                    st.markdown(
                                        f"**{test_item['test']}**"
                                    )
                                with t_col_sep1:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with t_col_result:
                                    st.markdown(test_item["result"])
                                with t_col_sep2:
                                    st.markdown(
                                        "<div style='border-left: 2px solid"
                                        " #ccc; height: 100%;"
                                        " min-height: 30px;'></div>",
                                        unsafe_allow_html=True,
                                    )
                                with t_col_normal:
                                    st.markdown(test_item["normal"])
                                with t_col_edit:
                                    if st.button(
                                        "\u270F\uFE0F",
                                        key=f"epen_lab_{real_idx}_{t_idx}",
                                    ):
                                        st.session_state.editing_lab_item = (
                                            f"elt_{real_idx}_{t_idx}"
                                        )
                                        st.rerun()
                                with t_col_trash:
                                    if st.button(
                                        "\U0001f5d1\uFE0F",
                                        key=f"etrash_lab_{real_idx}_{t_idx}",
                                    ):
                                        st.session_state[
                                            edit_tests_key
                                        ].pop(t_idx)
                                        st.rerun()

                    edit_add_cols = st.columns([2, 2, 2, 1])
                    with edit_add_cols[0]:
                        edit_new_test = st.text_input(
                            "Test",
                            key=f"edit_lab_newtest_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Test",
                        )
                    with edit_add_cols[1]:
                        edit_new_result = st.text_input(
                            "Result",
                            key=f"edit_lab_newresult_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Result",
                        )
                    with edit_add_cols[2]:
                        edit_new_normal = st.text_input(
                            "Normal",
                            key=f"edit_lab_newnormal_{real_idx}",
                            label_visibility="collapsed",
                            placeholder="Normal",
                        )
                    with edit_add_cols[3]:
                        if st.button(
                            "Add",
                            key=f"edit_lab_add_{real_idx}",
                            use_container_width=True,
                        ):
                            if (
                                edit_new_test.strip()
                                and edit_new_result.strip()
                                and edit_new_normal.strip()
                            ):
                                st.session_state[edit_tests_key].append({
                                    "test": edit_new_test.strip(),
                                    "result": edit_new_result.strip(),
                                    "normal": edit_new_normal.strip(),
                                })
                                st.rerun()

                    edited_lab_doctor = st.text_input(
                        "Doctor",
                        value=lt["doctor"],
                        key=f"edit_lab_doctor_{real_idx}",
                    )

                    edited_lab_notes = st.text_area(
                        "Notes (Markdown supported)",
                        value=lt.get("notes", ""),
                        height=120,
                        key=f"edit_lab_notes_{real_idx}",
                    )

                    if lt.get("attachment"):
                        st.caption(
                            f"Current attachment: "
                            f"{lt['attachment']['filename']}"
                        )
                    edit_lab_file = st.file_uploader(
                        "Replace PDF (optional)",
                        type=["pdf"],
                        key=f"edit_lab_file_{real_idx}",
                    )

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button(
                            "Save Changes",
                            key=f"save_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            updated_lab: dict[str, Any] = {
                                "date": edited_lab_date.strftime("%Y-%m-%d"),
                                "tests": list(
                                    st.session_state[edit_tests_key]
                                ),
                                "doctor": edited_lab_doctor.strip(),
                                "notes": edited_lab_notes.strip(),
                            }

                            if edit_lab_file is not None:
                                lab_attach_dir = (
                                    f"lab_attachments/{username}"
                                )
                                os.makedirs(lab_attach_dir, exist_ok=True)
                                lab_ts: str = datetime.now().strftime(
                                    "%Y%m%d_%H%M%S"
                                )
                                lab_sname: str = (
                                    f"{lab_ts}_{edit_lab_file.name}"
                                )
                                lab_fpath: str = os.path.join(
                                    lab_attach_dir, lab_sname
                                )
                                with open(lab_fpath, "wb") as f:
                                    f.write(edit_lab_file.getbuffer())
                                updated_lab["attachment"] = {
                                    "filename": edit_lab_file.name,
                                    "filepath": lab_fpath,
                                }
                                logger.info(
                                    "Lab attachment updated: %s",
                                    lab_fpath,
                                )
                            elif lt.get("attachment"):
                                updated_lab["attachment"] = lt["attachment"]

                            st.session_state.lab_tests[real_idx] = (
                                updated_lab
                            )
                            del st.session_state[edit_tests_key]
                            save_user_data(username)
                            st.session_state.editing_lab_test = None
                            logger.info("Lab test updated")
                            st.rerun()
                    with col_cancel_edit:
                        if st.button(
                            "Cancel",
                            key=f"cancel_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            if edit_tests_key in st.session_state:
                                del st.session_state[edit_tests_key]
                            st.session_state.editing_lab_test = None
                            st.rerun()
                else:
                    col_date, col_doctor, col_spacer = st.columns(
                        [2, 3, 1]
                    )

                    with col_date:
                        parsed_date = datetime.strptime(
                            lt["date"], "%Y-%m-%d"
                        )
                        display_date = (
                            f"{parsed_date.strftime('%B')} "
                            f"{parsed_date.day}, "
                            f"{parsed_date.year}"
                        )
                        st.markdown(f"**{display_date}**")
                    with col_doctor:
                        st.markdown(f"**Dr.** {lt['doctor']}")

                    # Test list
                    for test_item in lt.get("tests", []):
                        with st.container(border=True):
                            t_col_test, t_col_sep1, t_col_result, t_col_sep2, t_col_normal = (
                                st.columns([2, 0.15, 2, 0.15, 2])
                            )
                            with t_col_test:
                                st.markdown(
                                    f"**{test_item['test']}**"
                                )
                            with t_col_sep1:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with t_col_result:
                                st.markdown(test_item["result"])
                            with t_col_sep2:
                                st.markdown(
                                    "<div style='border-left: 2px solid"
                                    " #ccc; height: 100%;"
                                    " min-height: 30px;'></div>",
                                    unsafe_allow_html=True,
                                )
                            with t_col_normal:
                                st.markdown(test_item["normal"])

                    if lt.get("notes"):
                        st.markdown("---")
                        st.markdown(lt["notes"])

                    if lt.get("attachment"):
                        st.markdown("---")
                        st.caption(
                            f"📄 {lt['attachment']['filename']}"
                        )
                        with open(
                            lt["attachment"]["filepath"], "rb"
                        ) as f:
                            st.download_button(
                                label="Download PDF",
                                data=f,
                                file_name=lt["attachment"]["filename"],
                                mime="application/pdf",
                                key=f"dl_lab_{real_idx}",
                            )

                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(
                            "Edit",
                            key=f"edit_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_lab_test = real_idx
                            st.rerun()
                    with col_delete:
                        if st.button(
                            "Delete",
                            key=f"del_lab_{real_idx}",
                            use_container_width=True,
                        ):
                            st.session_state.lab_tests.pop(real_idx)
                            save_user_data(username)
                            logger.info("Lab test deleted")
                            st.rerun()
    else:
        st.info(
            "No lab tests yet. Start tracking your lab results!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: HEALTH JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption(
            "💡 Formatting: `**bold**`, `*italic*`, "
            "`- bullet point`, `1. numbered list`"
        )

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
