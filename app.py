"""
Health Assistant - Streamlit Application

Multi-user health assistant with tool-calling agent, document management,
and health journaling. Minimalist Nordic design.
"""

import os
from io import BytesIO
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage

from config import load_config, save_config
from auth import load_authenticator, render_login
from user_data import init_session_state, save_user_data
from tools import create_tools
from agent import bind_tools, run_agent
from vector_store import (
    get_or_create_vectorstore,
    get_retriever,
    add_pdf_to_vectorstore,
    remove_pdf_from_vectorstore,
    sync_vectorstore,
)


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
    """Load an external CSS file and inject it into the page."""
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(os.path.join(os.path.dirname(__file__), "assets", "style.css"))


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

config = load_config()

# Keep a session-state copy so that adds/deletes are reflected immediately
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = list(config["pdf_files"])
config["pdf_files"] = st.session_state.pdf_files


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
authenticator = load_authenticator()

if ("authentication_status" not in st.session_state
        or st.session_state["authentication_status"] is not True):
    render_login(authenticator)

# User is authenticated
name = st.session_state["name"]
username = st.session_state["username"]

init_session_state(username)


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE RESOURCES (CACHED)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def initialize_vectorstore():
    """Initialize vector store - runs once and caches the result."""
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
    return vectorstore, get_retriever(vectorstore, k=config["retriever"]["k"])


@st.cache_resource
def initialize_llm():
    """Initialize LLM - runs once and caches the result."""
    return ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
    )


with st.spinner("Loading..."):
    vectorstore, retriever = initialize_vectorstore()
    llm = initialize_llm()


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════
tools = create_tools(vectorstore, retriever, llm, config, SERPAPI_API_KEY)
llm_with_tools, tools_by_name = bind_tools(llm, tools)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: SIDEBAR CONFIG ITEM
# ═══════════════════════════════════════════════════════════════════════════════
def render_config_item(label: str, value) -> None:
    """Render a single config item in the sidebar."""
    st.markdown(f"""
    <div class="config-item">
        <div class="config-label">{label}</div>
        <div class="config-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown(f"**Welcome, {name}!**")

    try:
        authenticator.logout(location='sidebar', button_name='Sign Out')
    except TypeError:
        authenticator.logout('Sign Out', 'sidebar')

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
                st.markdown(f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #E8E8E8;">
                    <span style="color: #6B7280; font-size: 12px;
                           font-weight: 500;">{reminder['date']}</span><br>
                    <span style="color: #1F2937;
                           font-size: 13px;">{reminder['text']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_rem_{i}", help="Delete",
                             use_container_width=True):
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
    st.caption("Powered by OpenAI & LangChain")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Health Assistant")

tab1, tab2, tab3 = st.tabs(["Chat", "Manage Documents", "Journal"])


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

            st.markdown(f"""
            <div class="chat-question">
                <strong style="color: #1F2937;">Q:</strong> {qa['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="chat-answer">
                <strong style="color: #1F2937;">A:</strong><br><br>
                {qa['answer']}
            </div>
            """, unsafe_allow_html=True)

            if qa.get("tools_used"):
                tools_str = ", ".join(qa["tools_used"])
                st.caption(f"🔧 Tools used: {tools_str}")

            if qa.get("sources"):
                for i, source in enumerate(qa["sources"], 1):
                    if source == "Knowledge Base":
                        st.caption("📄 Source: Knowledge Base")
                    else:
                        st.caption(f"🔗 Source {i}: {source}")

        st.markdown("---")

    # Input area
    question = st.text_input(
        "Question",
        placeholder="e.g. What does my knowledge base say about blood pressure?",
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
        ask_button = st.button("Ask", key="ask_btn", use_container_width=True)

    with col_clear:
        if st.button("Clear Chat", key="clear_btn", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.agent_messages = []
            st.rerun()

    if ask_button:
        if question:
            with st.spinner("Thinking..."):
                ask_error = None
                try:
                    # If a PDF is attached, extract text and include it
                    user_message = question
                    if uploaded_lab:
                        reader = PdfReader(BytesIO(uploaded_lab.getbuffer()))
                        pdf_text = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        )
                        if pdf_text.strip():
                            user_message = (
                                f"{question}\n\n"
                                f"[Attached lab report text]\n{pdf_text}"
                            )
                        else:
                            ask_error = (
                                "Could not extract text from the uploaded PDF. "
                                "The file may be image-based."
                            )

                    if not ask_error:
                        msgs_before = len(st.session_state.agent_messages)

                        answer_text = run_agent(
                            user_message,
                            st.session_state.agent_messages,
                            llm_with_tools,
                            tools_by_name,
                        )

                        # Identify tools and extract sources
                        tools_used = []
                        sources = []
                        for msg in st.session_state.agent_messages[msgs_before:]:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc["name"] not in tools_used:
                                        tools_used.append(tc["name"])
                            if (isinstance(msg, ToolMessage)
                                    and "URL:" in msg.content):
                                for line in msg.content.split("\n"):
                                    if line.strip().startswith("URL:"):
                                        url = line.strip().replace(
                                            "URL:", ""
                                        ).strip()
                                        if url:
                                            sources.append(url)

                        if (not sources
                                and "search_knowledge_base" in tools_used):
                            sources = ["Knowledge Base"]

                        st.session_state.conversation_history.append({
                            "question": question,
                            "answer": answer_text,
                            "tools_used": tools_used,
                            "sources": sources,
                        })

                        st.session_state.question_counter += 1

                except Exception as e:
                    ask_error = str(e)

            if ask_error:
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
                filename = os.path.basename(pdf_path)
                file_size = "Unknown size"
                if os.path.exists(pdf_path):
                    size_bytes = os.path.getsize(pdf_path)
                    size_kb = size_bytes / 1024
                    if size_kb < 1024:
                        file_size = f"{size_kb:.1f} KB"
                    else:
                        file_size = f"{size_kb / 1024:.1f} MB"

                st.markdown(f"📄 **{filename}** · {file_size}")

            with col2:
                if st.button("Delete", key=f"del_pdf_{idx}",
                             use_container_width=True):
                    try:
                        remove_pdf_from_vectorstore(vectorstore, pdf_path)
                        st.session_state.pdf_files.remove(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")

                    st.rerun()
    else:
        st.info("No documents in knowledge base yet")

    st.markdown("---")

    st.markdown("#### Add New Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        key="kb_pdf_upload",
        help="Upload health-related PDF documents to add to your knowledge base",
    )

    if uploaded_pdf:
        st.success(f"✓ Selected: **{uploaded_pdf.name}**")

        if st.button("Add to Knowledge Base", type="primary",
                     use_container_width=True):
            add_success = False
            with st.spinner("Adding document to knowledge base..."):
                try:
                    pdf_dir = "pdfs"
                    os.makedirs(pdf_dir, exist_ok=True)

                    pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)

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
                except Exception as e:
                    st.error(f"❌ Error adding document: {str(e)}")

            if add_success:
                st.rerun()
    else:
        st.info("👆 Choose a PDF file to add to your knowledge base")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: HEALTH JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")

    col1, col2 = st.columns([3, 1])

    with col1:
        journal_title = st.text_input(
            "Title",
            placeholder="Entry title...",
            key=f"journal_title_{st.session_state.journal_form_key}",
        )

        journal_entry = st.text_area(
            "Entry",
            placeholder="How are you feeling today?",
            height=120,
            key=f"journal_entry_{st.session_state.journal_form_key}",
        )

        uploaded_file = st.file_uploader(
            "Attachment (optional)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif'],
            key=f"journal_file_{st.session_state.file_uploader_key}",
        )

    with col2:
        journal_date = st.date_input("Date", key="journal_date")

        st.markdown("")

        if st.button("Save Entry", use_container_width=True):
            if journal_title and journal_entry:
                entry_data = {
                    "title": journal_title,
                    "date": journal_date.strftime("%Y-%m-%d"),
                    "entry": journal_entry,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                if uploaded_file is not None:
                    attachment_dir = f"journal_attachments/{username}"
                    os.makedirs(attachment_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = uploaded_file.name.rsplit('.', 1)[-1].lower()
                    safe_filename = f"{timestamp}_{uploaded_file.name}"
                    file_path = os.path.join(attachment_dir, safe_filename)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    entry_data["attachment"] = {
                        "filename": uploaded_file.name,
                        "filepath": file_path,
                        "type": file_extension,
                    }

                st.session_state.journal_entries.append(entry_data)
                save_user_data(username)
                st.session_state.file_uploader_key += 1
                st.session_state.journal_form_key += 1
                st.rerun()
            else:
                st.warning("Please enter both title and entry")

    st.markdown("---")

    # Past Entries
    if st.session_state.journal_entries:
        st.markdown("#### Past Entries")

        for entry in reversed(st.session_state.journal_entries):
            attachment_icon = " 📎" if "attachment" in entry else ""
            is_editing = (
                st.session_state.editing_entry == entry['timestamp']
            )

            with st.container(border=True):
                if is_editing:
                    st.markdown("**✏️ Editing Entry**")

                    edited_title = st.text_input(
                        "Title",
                        value=entry.get('title', ''),
                        key=f"edit_title_{entry['timestamp']}",
                    )

                    edited_entry_text = st.text_area(
                        "Entry",
                        value=entry['entry'],
                        height=120,
                        key=f"edit_entry_{entry['timestamp']}",
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("Save Changes",
                                     key=f"save_{entry['timestamp']}",
                                     use_container_width=True):
                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e['timestamp'] == entry['timestamp']:
                                    st.session_state.journal_entries[idx][
                                        'title'] = edited_title
                                    st.session_state.journal_entries[idx][
                                        'entry'] = edited_entry_text
                                    break
                            save_user_data(username)
                            st.session_state.editing_entry = None
                            st.success("Entry updated!")
                            st.rerun()

                    with col_cancel:
                        if st.button("Cancel",
                                     key=f"cancel_{entry['timestamp']}",
                                     use_container_width=True):
                            st.session_state.editing_entry = None
                            st.rerun()
                else:
                    st.markdown(
                        f"**{entry['date']} — "
                        f"{entry.get('title', 'Untitled')}"
                        f"{attachment_icon}**"
                    )
                    st.markdown(entry['entry'])

                    if "attachment" in entry:
                        st.markdown("---")
                        attachment = entry["attachment"]
                        file_type = attachment["type"]

                        if file_type in ['png', 'jpg', 'jpeg', 'gif']:
                            st.image(
                                attachment["filepath"],
                                caption=attachment["filename"],
                                use_container_width=True,
                            )
                        elif file_type == 'pdf':
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
                        if st.button("Edit Entry",
                                     key=f"edit_{entry['timestamp']}",
                                     use_container_width=True):
                            st.session_state.editing_entry = entry['timestamp']
                            st.rerun()

                    with col_delete:
                        if st.button("Delete Entry",
                                     key=f"del_{entry['timestamp']}",
                                     use_container_width=True):
                            if ("attachment" in entry
                                    and "filepath" in entry["attachment"]):
                                filepath = os.path.normpath(
                                    entry["attachment"]["filepath"]
                                )
                                if os.path.exists(filepath):
                                    try:
                                        os.remove(filepath)
                                    except Exception:
                                        pass

                            for idx, e in enumerate(
                                st.session_state.journal_entries
                            ):
                                if e['timestamp'] == entry['timestamp']:
                                    st.session_state.journal_entries.pop(idx)
                                    break

                            save_user_data(username)
                            st.rerun()
    else:
        st.info(
            "No journal entries yet. Start tracking your health journey!"
        )
