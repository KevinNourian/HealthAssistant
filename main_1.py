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
from yaml.loader import SafeLoader
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from serpapi import GoogleSearch

from vector_store import get_or_create_vectorstore, get_retriever


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
# MINIMALIST NORDIC STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables */
    :root {
        --white: #FFFFFF;
        --off-white: #FAFAFA;
        --light-gray: #F5F5F5;
        --border-gray: #E8E8E8;
        --medium-gray: #D0D0D0;
        --text-gray: #6B7280;
        --text-dark: #1F2937;
        --text-black: #111827;
        --success: #10B981;
        --success-light: #D1FAE5;
        --warning: #F59E0B;
        --warning-light: #FEF3C7;
        --error: #EF4444;
        --error-light: #FEE2E2;
        --info-light: #F3F4F6;
        --accent-blue: #3B82F6;
        --accent-blue-light: #DBEAFE;
        --radius-md: 8px;
        --radius-lg: 12px;
    }
    
    /* Main app background */
    .stApp {
        background-color: var(--off-white);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: var(--text-black) !important;
    }
    
    h2, h3 {
        font-weight: 600 !important;
        color: var(--text-dark) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--white) !important;
        border-right: 1px solid var(--border-gray) !important;
    }
    
    /* ALL BUTTONS - Uniform white styling */
    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.5rem 1rem !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-gray) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        padding: 0 24px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: var(--text-gray) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-dark) !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-dark) !important;
        border-bottom: 2px solid var(--text-dark) !important;
    }
    
    /* Form Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stDateInput input {
        background-color: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.875rem !important;
        color: var(--text-dark) !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stDateInput input:focus {
        border-color: var(--medium-gray) !important;
        box-shadow: 0 0 0 3px var(--light-gray) !important;
    }
    
    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--white) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: var(--white) !important;
        border: 1px dashed var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.5rem !important;
    }
    
    /* Alerts */
    .stSuccess {
        background-color: var(--success-light) !important;
        border-left: 3px solid var(--success) !important;
    }
    
    .stInfo {
        background-color: var(--info-light) !important;
        border-left: 3px solid var(--text-gray) !important;
    }
    
    .stWarning {
        background-color: var(--warning-light) !important;
        border-left: 3px solid var(--warning) !important;
    }
    
    .stError {
        background-color: var(--error-light) !important;
        border-left: 3px solid var(--error) !important;
    }
    
    /* Containers with border */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-lg) !important;
    }
    
    /* Chat message styling */
    .chat-question {
        background: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        padding: 1rem !important;
        border-radius: var(--radius-md) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .chat-answer {
        background: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        padding: 1rem !important;
        border-radius: var(--radius-md) !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Hide form hints */
    .stForm small,
    [data-testid="stForm"] small,
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* Config display */
    .config-item {
        margin-bottom: 0.75rem;
    }
    
    .config-label {
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6B7280;
        margin-bottom: 0.125rem;
    }
    
    .config-value {
        font-size: 0.75rem;
        font-weight: 600;
        color: #1F2937;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SETUP
# ═══════════════════════════════════════════════════════════════════════════════
with open('credentials.yaml') as file:
    config_auth = yaml.load(file, Loader=SafeLoader)

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
    st.session_state.pdf_summary = ""
    st.session_state.lab_analysis = ""
    st.session_state.lab_error = None
    st.session_state.editing_entry = None
    st.session_state.conversation_history = []
    st.session_state.question_counter = 0

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
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
@st.cache_data
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


config = load_config()


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
    """Generate a summary of a PDF."""
    try:
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
{combined_text[:3000]}

Summary:"""
        
        response = llm.invoke(summary_prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


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
    # User greeting
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
        st.markdown(f"""
        <div class="config-item">
            <div class="config-label">Model</div>
            <div class="config-value">{config["llm"]["model"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="config-item">
            <div class="config-label">Chunks</div>
            <div class="config-value">{config["chunking"]["chunk_size"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="config-item">
            <div class="config-label">Temp</div>
            <div class="config-value">{config["llm"]["temperature"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="config-item">
            <div class="config-label">Top-K</div>
            <div class="config-value">{config["retriever"]["k"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Powered by OpenAI & LangChain")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Health Assistant")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Ask Question",
    "Summarize",
    "Analyse Lab Report",
    "Journal"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: ASK HEALTH QUESTION (CHAT STYLE)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a Health Question")
    st.caption("Get answers from your knowledge base or the web")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        for qa in st.session_state.conversation_history:
            # Question
            st.markdown(f"""
            <div class="chat-question">
                <strong style="color: #1F2937;">Q:</strong> {qa['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Answer
            st.markdown(f"""
            <div class="chat-answer">
                <strong style="color: #1F2937;">A:</strong><br><br>
                {qa['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources if available
            if qa.get('sources'):
                for source_label, source_url in qa['sources']:
                    if source_url != "Knowledge Base":
                        st.caption(f"  {source_label}: {source_url}")
                    else:
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
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
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
                    from pypdf import PdfReader
                    from io import BytesIO
                    
                    pdf_file = BytesIO(uploaded_lab_pdf.read())
                    pdf_reader = PdfReader(pdf_file)
                    
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
{pdf_text[:4000]}

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
# TAB 4: HEALTH JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Health Journal")
    st.caption("Track your health journey with notes and attachments")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        journal_title = st.text_input(
            "Title",
            placeholder="Entry title...",
            key="journal_title"
        )
        
        journal_entry = st.text_area(
            "Entry",
            placeholder="How are you feeling today?",
            height=120,
            key="journal_entry"
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
                    file_extension = uploaded_file.name.split('.')[-1]
                    safe_filename = f"{timestamp}_{uploaded_file.name}"
                    file_path = os.path.join(attachment_dir, safe_filename)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    entry_data["attachment"] = {
                        "filename": uploaded_file.name,
                        "filepath": file_path,
                        "type": file_extension.lower()
                    }
                
                st.session_state.journal_entries.append(entry_data)
                save_user_data()
                st.session_state.file_uploader_key += 1
                st.session_state.journal_title = ""
                st.session_state.journal_entry = ""
                st.success("Entry saved!")
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
