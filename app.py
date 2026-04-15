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
* ``ui/``               – individual tab and sidebar renderers
"""

import logging
import os
import sqlite3
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import load_config, validate_api_keys
from core.auth import load_authenticator, render_login
from core.rate_limiter import init_rate_limiter
from core.user_data import init_session_state
from core.tools import create_tools
from core.agent import build_graph
from core.vector_store import (
    get_or_create_vectorstore,
    get_hybrid_retriever,
    sync_vectorstore,
)

from ui import (
    sidebar,
    chat,
    documents,
    blood_pressure,
    medical_visit,
    vaccination,
    prescription,
    lab_test,
    journal,
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
# ════════════════════════════════��══════════════════════════════════════════════
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


# ═════════════════════════════════��═════════════════════════════════════════════
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
# ═══════════════════════════════��═══════════════════════════════════════════════
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
# ══════════════════════════════════════════════���═══════════════════════════���════
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
    init_rate_limiter("checkpoints.db")


# ═══════════════════════════════════════════��═══════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════���═══════════════════════════════════
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


# ═════════════════════════════════════════════════���═════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════���═════════════════════════════
sidebar.render(username, name, authenticator, config)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Health Assistant")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Chat", "Manage Documents", "Blood Pressure", "Medical Visit",
    "Vaccination", "Prescription", "Lab Test", "Journal",
])

with tab1:
    chat.render(username, graph, llm, config)

with tab2:
    documents.render(username, config, vectorstore)

with tab3:
    blood_pressure.render(username)

with tab4:
    medical_visit.render(username)

with tab5:
    vaccination.render(username)

with tab6:
    prescription.render(username)

with tab7:
    lab_test.render(username)

with tab8:
    journal.render(username)
