"""
Health Assistant - Streamlit App with Authentication
Multi-user app with login and user-specific data
Minimalist Nordic Design
"""

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
import os
import json
import yaml
from io import BytesIO
from yaml.loader import SafeLoader
from datetime import datetime

from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from serpapi import GoogleSearch

from vector_store import (
    get_or_create_vectorstore,
    get_retriever,
    add_pdf_to_vectorstore,
    remove_pdf_from_vectorstore,
    sync_vectorstore,
    normalize_source_path,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SUMMARY_MAX_CHARS = 3000
LAB_REPORT_MAX_CHARS = 4000


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
# LOAD ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════
try:
    with open('credentials.yaml') as file:
        config_auth = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Missing `credentials.yaml`. Please create the file with valid credentials.")
    st.stop()

authenticator = stauth.Authenticate(
    config_auth['credentials'],
    config_auth['cookie']['name'],
    config_auth['cookie']['key'],
    config_auth['cookie']['expiry_days']
)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION CHECK
# ═══════════════════════════════════════════════════════════════════════════════
if "authentication_status" not in st.session_state or st.session_state["authentication_status"] != True:
    with st.sidebar:
        st.markdown("## 🏥 Health Assistant")
        st.markdown("---")
        st.markdown("##### Sign In")

        # Use authenticator's built-in login
        authenticator.login(location='sidebar')

        if st.session_state.get("authentication_status") == False:
            st.error('Invalid username or password')
        elif st.session_state.get("authentication_status") == None:
            st.markdown("")
            st.markdown("""
            <div style="background: #F5F5F5; padding: 12px; border-radius: 8px; font-size: 13px;">
                <strong>Demo Accounts</strong><br>
                <span style="color: #6B7280;">alice / temp123</span><br>
                <span style="color: #6B7280;">bob / temp456</span>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.get("authentication_status") != True:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">Health Assistant</h1>
            <p style="color: #6B7280; font-size: 1.125rem;">Your personal health companion</p>
            <p style="color: #9CA3AF; font-size: 0.875rem; margin-top: 2rem;">← Please sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.rerun()

# User is authenticated
name = st.session_state["name"]
username = st.session_state["username"]
authentication_status = st.session_state["authentication_status"]


# ═══════════════════════════════════════════════════════════════════════════════
# USER DATA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
USER_DATA_FILE = f"user_data_{username}.json"


def load_user_data():
    """Load user-specific data from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {"reminders": [], "journal_entries": []}


def save_user_data():
    """Save user-specific data to JSON file"""
    data = {
        "reminders": st.session_state.reminders,
        "journal_entries": st.session_state.journal_entries
    }
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# Initialize session state
if "current_user" not in st.session_state or st.session_state.current_user != username:
    user_data = load_user_data()
    st.session_state.reminders = user_data.get("reminders", [])
    st.session_state.journal_entries = user_data.get("journal_entries", [])
    st.session_state.current_user = username
    st.session_state.file_uploader_key = 0
    st.session_state.journal_form_key = 0
    st.session_state.pdf_summary = ""
    st.session_state.lab_analysis = ""
    st.session_state.lab_error = None
    st.session_state.editing_entry = None
    st.session_state.conversation_history = []
    st.session_state.question_counter = 0

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "journal_form_key" not in st.session_state:
    st.session_state.journal_form_key = 0
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""
if "lab_analysis" not in st.session_state:
    st.session_state.lab_analysis = ""
if "lab_error" not in st.session_state:
    st.session_state.lab_error = None
if "editing_entry" not in st.session_state:
    st.session_state.editing_entry = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "question_counter" not in st.session_state:
    st.session_state.question_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config_data: dict, config_path: str = "config.json") -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


config = load_config()

# Keep a session-state copy so that adds/deletes are reflected immediately
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = list(config["pdf_files"])
config["pdf_files"] = st.session_state.pdf_files


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
        force_recreate=False
    )
    # Sync with current config (handles missed adds/deletes from crashes, etc.)
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
        temperature=config["llm"]["temperature"]
    )


with st.spinner("Loading..."):
    vectorstore, retriever = initialize_vectorstore()
    llm = initialize_llm()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG CHAIN SETUP
# ═══════════════════════════════════════════════════════════════════════════════
prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def serpapi_search(query: str, max_results: int = 3) -> list:
    """Search the web using SerpAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }

    try:
        search = GoogleSearch(params)
        result = search.get_dict()
        results = []

        if "organic_results" in result:
            for item in result["organic_results"][:max_results]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if title and snippet:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": link
                    })

        return results
    except Exception:
        return []


def summarize_pdf(pdf_path: str) -> str:
    """Generate a summary of a PDF using the vectorstore."""
    try:
        # Normalize the path to match what Chroma stores in metadata
        normalized = normalize_source_path(pdf_path)
        docs = vectorstore.similarity_search(
            "summary of document",
            k=10,
            filter={"source": normalized}
        )

        # Fallback: try the raw path in case it was stored un-normalized
        if not docs:
            docs = vectorstore.similarity_search(
                "summary of document",
                k=10,
                filter={"source": pdf_path}
            )

        if not docs:
            return f"No content found for {os.path.basename(pdf_path)}"

        combined_text = "\n\n".join([doc.page_content for doc in docs])

        summary_prompt = f"""Provide a comprehensive summary of the following health document.
Include main topics, key points, and important information.

Document content:
{combined_text[:SUMMARY_MAX_CHARS]}

Summary:"""

        response = llm.invoke(summary_prompt)
        return response.content

    except Exception as e:
        return f"Error generating summary: {str(e)}"


def render_config_item(label: str, value) -> None:
    """Render a single config item in the sidebar."""
    st.markdown(f"""
    <div class="config-item">
        <div class="config-label">{label}</div>
        <div class="config-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION GUARD
# ═══════════════════════════════════════════════════════════════════════════════
if authentication_status is not True:
    st.error("Authentication required. Please refresh the page.")
    st.stop()


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

    # ─────────────────────────────────────────────────────────────────────
    # Health Reminders
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### ⏰ Reminders")

    with st.form("reminder_form", clear_on_submit=True):
        reminder_text = st.text_input(
            "Reminder",
            placeholder="e.g., Doctor visit",
            label_visibility="collapsed"
        )
        reminder_date = st.date_input("Date", label_visibility="collapsed")

        if st.form_submit_button("Add Reminder", use_container_width=True):
            if reminder_text:
                st.session_state.reminders.append({
                    "text": reminder_text,
                    "date": reminder_date.strftime("%Y-%m-%d"),
                    "id": len(st.session_state.reminders)
                })
                save_user_data()
                st.rerun()

    if st.session_state.reminders:
        for i, reminder in enumerate(st.session_state.reminders):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div style="padding: 8px 0; border-bottom: 1px solid #E8E8E8;">
                    <span style="color: #6B7280; font-size: 12px; font-weight: 500;">{reminder['date']}</span><br>
                    <span style="color: #1F2937; font-size: 13px;">{reminder['text']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_rem_{i}", help="Delete", use_container_width=True):
                    st.session_state.reminders.pop(i)
                    save_user_data()
                    st.rerun()
    else:
        st.caption("No reminders yet")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # Knowledge Base
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 📚 Documents")

    if config["pdf_files"]:
        for pdf in config["pdf_files"]:
            st.caption(f"• {os.path.basename(pdf)}")
    else:
        st.caption("No documents loaded")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────
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

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ask Question",
    "Summarize",
    "Analyse Lab Report",
    "Manage Documents",
    "Journal"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: ASK HEALTH QUESTION (CHAT STYLE)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a Health Question")
    st.caption("Get answers from your knowledge base or the web")

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

            # Sources if available
            if qa.get('sources'):
                for source_label, source_url in qa['sources']:
                    st.caption(f"  {source_label}: {source_url}")

        st.markdown("---")

    # Input for new question (full width)
    question = st.text_input(
        "Question",
        placeholder="Type your health question here...",
        key=f"health_question_{st.session_state.question_counter}",
        label_visibility="collapsed"
    )

    # Buttons side by side below input
    col_ask, col_clear = st.columns(2)

    with col_ask:
        ask_button = st.button("Ask", key="ask_btn", use_container_width=True)

    with col_clear:
        if st.button("Clear Chat", key="clear_btn", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

    if ask_button:
        if question:
            with st.spinner("Searching..."):
                ask_error = None
                try:
                    response = chain.invoke(question)
                    answer_text = response.content.strip()
                    sources = []

                    if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                        web_results = serpapi_search(question)

                        if web_results:
                            combined_answer = "Here's what I found:\n\n"
                            for result in web_results:
                                combined_answer += f"**{result['title']}**\n{result['snippet']}\n\n"

                            answer_text = combined_answer
                            sources = [
                                (f"Source {i}", result['url'])
                                for i, result in enumerate(web_results, 1)
                            ]
                        else:
                            answer_text = "I couldn't find enough information to answer this question."
                            sources = []
                    else:
                        sources = [("Source", "Knowledge Base")]

                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'answer': answer_text,
                        'sources': sources
                    })

                    # Increment counter to clear input box
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
# TAB 2: SUMMARIZE PDF
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Summarize Document")
    st.caption("Generate AI summaries of your health documents")

    if config["pdf_files"]:
        selected_pdf = st.selectbox(
            "Select document",
            options=config["pdf_files"],
            format_func=lambda x: os.path.basename(x),
            key="pdf_select"
        )

        if st.button("Summarize", key="summarize_btn"):
            with st.spinner("Generating summary..."):
                summary = summarize_pdf(selected_pdf)
                st.session_state.pdf_summary = summary

        # Display summary
        if st.session_state.pdf_summary:
            st.markdown("#### Summary")
            with st.container(border=True):
                st.markdown(st.session_state.pdf_summary)
    else:
        st.info("No documents available to summarize")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ANALYSE LAB REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Analyse Lab Report")
    st.caption("Upload your blood work or lab results for AI analysis")

    uploaded_lab_pdf = st.file_uploader(
        "Upload Lab Report",
        type=['pdf'],
        key="lab_pdf_upload",
        help="Upload your blood work, lab results, or medical test report"
    )

    if uploaded_lab_pdf is not None:
        st.success(f"Uploaded: {uploaded_lab_pdf.name}")

        if st.button("Analyse", key="analyse_btn"):
            with st.spinner("Analyzing report..."):
                try:
                    pdf_reader = PdfReader(BytesIO(uploaded_lab_pdf.getbuffer()))

                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text() + "\n"

                    if not pdf_text.strip():
                        st.session_state.lab_analysis = None
                        st.session_state.lab_error = "Could not extract text from PDF. The file may be image-based."
                    else:
                        analysis_prompt = f"""You are a medical AI assistant analyzing lab results.

Please analyze the following lab report and provide:

1. **Key Findings**: List the main test results with their values
2. **Normal vs. Abnormal**: Identify which values are outside normal ranges
3. **Health Implications**: Explain what the results might indicate
4. **Recommendations**: Suggest next steps

IMPORTANT: This is for informational purposes only. Always recommend consulting with a healthcare provider.

Lab Report:
{pdf_text[:LAB_REPORT_MAX_CHARS]}

Analysis:"""

                        analysis_response = llm.invoke(analysis_prompt)
                        st.session_state.lab_analysis = analysis_response.content
                        st.session_state.lab_error = None

                except Exception as e:
                    st.session_state.lab_analysis = None
                    st.session_state.lab_error = f"Error analyzing PDF: {str(e)}"

        # Display results
        if st.session_state.lab_error:
            st.error(st.session_state.lab_error)

        if st.session_state.lab_analysis:
            st.markdown("#### Analysis Results")
            with st.container(border=True):
                st.markdown(st.session_state.lab_analysis)

            st.warning("""**⚠️ Medical Disclaimer**
This analysis is for informational purposes only and should NOT be considered medical advice. Always consult with a qualified healthcare provider to interpret your lab results.""")
    else:
        st.info("Upload a PDF of your lab report to get started")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MANAGE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Manage Knowledge Base")
    st.caption("Upload PDFs to expand your health knowledge base")

    # ─────────────────────────────────────────────────────────────────────
    # Current Documents Section
    # ─────────────────────────────────────────────────────────────────────
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
                        file_size = f"{size_kb/1024:.1f} MB"

                st.markdown(f"📄 **{filename}** · {file_size}")

            with col2:
                if st.button("Delete", key=f"del_pdf_{idx}", use_container_width=True):
                    try:
                        # 1. Remove embeddings from the vectorstore
                        remove_pdf_from_vectorstore(vectorstore, pdf_path)

                        # 2. Remove from config and session state
                        st.session_state.pdf_files.remove(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                        # 3. Delete physical file
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                        # 4. Clear vectorstore cache so retriever is recreated
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")

                    st.rerun()
    else:
        st.info("No documents in knowledge base yet")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # Upload New Document Section
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("#### Add New Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        key="kb_pdf_upload",
        help="Upload health-related PDF documents to add to your knowledge base"
    )

    if uploaded_pdf:
        st.success(f"✓ Selected: **{uploaded_pdf.name}**")

        if st.button("Add to Knowledge Base", type="primary", use_container_width=True):
            add_success = False
            with st.spinner("Adding document to knowledge base..."):
                try:
                    # 1. Create pdfs directory if it doesn't exist
                    pdf_dir = "pdfs"
                    os.makedirs(pdf_dir, exist_ok=True)

                    # 2. Save PDF file to disk
                    pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    # 3. Update config and session state (avoid duplicates)
                    if pdf_path not in st.session_state.pdf_files:
                        st.session_state.pdf_files.append(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                    # 4. Embed the new document into the vectorstore
                    add_pdf_to_vectorstore(
                        vectorstore,
                        pdf_path,
                        chunk_size=config["chunking"]["chunk_size"],
                        chunk_overlap=config["chunking"]["chunk_overlap"],
                    )

                    # 5. Clear vectorstore cache so retriever is recreated
                    st.cache_resource.clear()

                    add_success = True
                except Exception as e:
                    st.error(f"❌ Error adding document: {str(e)}")

            if add_success:
                st.rerun()
    else:
        st.info("👆 Choose a PDF file to add to your knowledge base")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: HEALTH JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")

    col1, col2 = st.columns([3, 1])

    with col1:
        journal_title = st.text_input(
            "Title",
            placeholder="Entry title...",
            key=f"journal_title_{st.session_state.journal_form_key}"
        )

        journal_entry = st.text_area(
            "Entry",
            placeholder="How are you feeling today?",
            height=120,
            key=f"journal_entry_{st.session_state.journal_form_key}"
        )

        uploaded_file = st.file_uploader(
            "Attachment (optional)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif'],
            key=f"journal_file_{st.session_state.file_uploader_key}"
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
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                        "type": file_extension
                    }

                st.session_state.journal_entries.append(entry_data)
                save_user_data()
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
            is_editing = (st.session_state.editing_entry == entry['timestamp'])

            with st.container(border=True):
                if is_editing:
                    # Edit mode
                    st.markdown("**✏️ Editing Entry**")

                    edited_title = st.text_input(
                        "Title",
                        value=entry.get('title', ''),
                        key=f"edit_title_{entry['timestamp']}"
                    )

                    edited_entry_text = st.text_area(
                        "Entry",
                        value=entry['entry'],
                        height=120,
                        key=f"edit_entry_{entry['timestamp']}"
                    )

                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("Save Changes", key=f"save_{entry['timestamp']}", use_container_width=True):
                            for idx, e in enumerate(st.session_state.journal_entries):
                                if e['timestamp'] == entry['timestamp']:
                                    st.session_state.journal_entries[idx]['title'] = edited_title
                                    st.session_state.journal_entries[idx]['entry'] = edited_entry_text
                                    break
                            save_user_data()
                            st.session_state.editing_entry = None
                            st.success("Entry updated!")
                            st.rerun()

                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_{entry['timestamp']}", use_container_width=True):
                            st.session_state.editing_entry = None
                            st.rerun()
                else:
                    # Display mode
                    st.markdown(f"**{entry['date']} — {entry.get('title', 'Untitled')}{attachment_icon}**")
                    st.markdown(entry['entry'])

                    if "attachment" in entry:
                        st.markdown("---")
                        attachment = entry["attachment"]
                        file_type = attachment["type"]

                        if file_type in ['png', 'jpg', 'jpeg', 'gif']:
                            st.image(attachment["filepath"], caption=attachment["filename"], use_container_width=True)
                        elif file_type == 'pdf':
                            st.caption(f"📄 {attachment['filename']}")
                            with open(attachment["filepath"], "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f,
                                    file_name=attachment["filename"],
                                    mime="application/pdf",
                                    key=f"dl_{entry['timestamp']}"
                                )

                    # Edit and Delete buttons
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button("Edit Entry", key=f"edit_{entry['timestamp']}", use_container_width=True):
                            st.session_state.editing_entry = entry['timestamp']
                            st.rerun()

                    with col_delete:
                        if st.button("Delete Entry", key=f"del_{entry['timestamp']}", use_container_width=True):
                            if "attachment" in entry and "filepath" in entry["attachment"]:
                                filepath = os.path.normpath(entry["attachment"]["filepath"])
                                if os.path.exists(filepath):
                                    try:
                                        os.remove(filepath)
                                    except Exception:
                                        pass

                            for idx, e in enumerate(st.session_state.journal_entries):
                                if e['timestamp'] == entry['timestamp']:
                                    st.session_state.journal_entries.pop(idx)
                                    break

                            save_user_data()
                            st.rerun()
    else:
        st.info("No journal entries yet. Start tracking your health journey!")
