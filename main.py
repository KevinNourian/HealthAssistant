# main.py

"""
Health Assistant - Streamlit App with Authentication
Multi-user app with login and user-specific data
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


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide form submission hints */
    .stForm [data-testid="stFormSubmitContent"] {
        display: none !important;
    }
    [data-testid="stFormSubmitButton"] ~ div {
        display: none !important;
    }
    .stFormSubmitContent {
        display: none !important;
    }
    [data-testid="stForm"] small {
        display: none !important;
    }
    .stTextArea [data-testid="stMarkdownContainer"] small {
        display: none !important;
    }
    .stTextArea small {
        display: none !important;
    }
    
    /* Make tabs bigger and bolder */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0px 24px;
        font-size: 18px;
        font-weight: 700;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# -------------------------
# Authentication Setup
# -------------------------
# Load credentials
with open('credentials.yaml') as file:
    config_auth = yaml.load(file, Loader=SafeLoader)

# Create authenticator object
authenticator = stauth.Authenticate(
    config_auth['credentials'],
    config_auth['cookie']['name'],
    config_auth['cookie']['key'],
    config_auth['cookie']['expiry_days']
)


# -------------------------
# Authentication Check
# -------------------------
# Check if user is already authenticated
if "authentication_status" not in st.session_state or st.session_state["authentication_status"] != True:
    # User not logged in - show login in sidebar
    with st.sidebar:
        st.title("🏥 Health Assistant")
        st.markdown("### Login")
        
        # Show login form in sidebar
        authenticator.login(location='sidebar')
        
        # Check authentication status
        if st.session_state.get("authentication_status") == False:
            st.error('Incorrect credentials')
        elif st.session_state.get("authentication_status") == None:
            st.info("**Demo:**\nalice/temp123\nbob/temp456")
    
    # If not authenticated, show message in main and stop
    if st.session_state.get("authentication_status") != True:
        st.title("Welcome to Health Assistant")
        st.info("👈 Please login using the sidebar")
        st.stop()
    
    # Just logged in - rerun to show app
    st.rerun()

# User is authenticated
name = st.session_state["name"]
username = st.session_state["username"]
authentication_status = st.session_state["authentication_status"]


# -------------------------
# User is authenticated - Load their data
# -------------------------

# User-specific data file
USER_DATA_FILE = f"user_data_{username}.json"


def load_user_data():
    """Load user-specific data from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {
        "reminders": [],
        "journal_entries": []
    }


def save_user_data():
    """Save user-specific data to JSON file"""
    data = {
        "reminders": st.session_state.reminders,
        "journal_entries": st.session_state.journal_entries
    }
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# Initialize session state with user data
# IMPORTANT: Reload data if user has changed (fixes data isolation bug)
if "current_user" not in st.session_state or st.session_state.current_user != username:
    user_data = load_user_data()
    st.session_state.reminders = user_data.get("reminders", [])
    st.session_state.journal_entries = user_data.get("journal_entries", [])
    st.session_state.current_user = username
    st.session_state.answer = ""
    st.session_state.sources = []
    st.session_state.file_uploader_key = 0  # For clearing file uploader

if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


# -------------------------
# Load Configuration
# -------------------------
@st.cache_data
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


config = load_config()


# -------------------------
# Initialize Vector Store (Cached)
# -------------------------
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


# -------------------------
# Initialize LLM (Cached)
# -------------------------
@st.cache_resource
def initialize_llm():
    """Initialize LLM - runs once and caches the result."""
    return ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"]
    )


# -------------------------
# Load Resources
# -------------------------
with st.spinner("🔄 Loading knowledge base..."):
    vectorstore, retriever = initialize_vectorstore()
    llm = initialize_llm()


# -------------------------
# Define RAG Prompt
# -------------------------
prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)


# -------------------------
# Create RAG Chain
# -------------------------
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# -------------------------
# Web Search Function
# -------------------------
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
    except Exception as e:
        return []


# -------------------------
# PDF Summary Function
# -------------------------
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


# -------------------------
# ONLY CONTINUE IF AUTHENTICATED
# -------------------------
if authentication_status is not True:
    # This should never be reached, but as a safeguard
    st.error("Authentication required. Please refresh the page.")
    st.stop()


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    # User info and logout
    st.write(f"### Welcome, {name}! 👋")
    try:
        # Try new API (streamlit-authenticator 0.3.0+)
        authenticator.logout(location='sidebar', button_name='Logout')
    except TypeError:
        # Fallback to old API
        authenticator.logout('Logout', 'sidebar')
    
    st.divider()
    
    # ============================================
    # HEALTH REMINDERS
    # ============================================
    st.header("⏰ Health Reminders")
    
    with st.form("reminder_form", clear_on_submit=True):
        reminder_text = st.text_input("Reminder", placeholder="Doctor visit", key="reminder_input")
        reminder_date = st.date_input("Date", key="reminder_date")
        
        add_reminder = st.form_submit_button("➕ Add", use_container_width=True)
        
        if add_reminder and reminder_text:
            st.session_state.reminders.append({
                "text": reminder_text,
                "date": reminder_date.strftime("%Y-%m-%d"),
                "id": len(st.session_state.reminders)
            })
            save_user_data()  # Save to file
            st.success("✅ Reminder added!")
            st.rerun()
    
    if st.session_state.reminders:
        st.subheader("Active Reminders:")
        for i, reminder in enumerate(st.session_state.reminders):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📅 **{reminder['date']}** - {reminder['text']}")
            with col2:
                if st.button("❌", key=f"delete_reminder_{i}"):
                    st.session_state.reminders.pop(i)
                    save_user_data()  # Save to file
                    st.rerun()
    else:
        st.info("No active reminders")
    
    st.divider()
    
    # ============================================
    # KNOWLEDGE BASE
    # ============================================
    st.header("📚 Knowledge Base")
    
    st.subheader("Loaded Documents")
    if config["pdf_files"]:
        for i, pdf in enumerate(config["pdf_files"], 1):
            st.write(f"{i}. {os.path.basename(pdf)}")
    else:
        st.write("No PDFs loaded")
    
    st.divider()
    
    # ============================================
    # CONFIGURATION
    # ============================================
    st.header("⚙️ Configuration")
    
    st.markdown("""
    <style>
    .config-metric {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .config-label {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 0.1rem;
    }
    .config-value {
        font-size: 0.95rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="config-metric">
            <div class="config-label">Model</div>
            <div class="config-value">{config["llm"]["model"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="config-metric">
            <div class="config-label">Chunk Size</div>
            <div class="config-value">{config["chunking"]["chunk_size"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="config-metric">
            <div class="config-label">Temperature</div>
            <div class="config-value">{config["llm"]["temperature"]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown(f"""
        <div class="config-metric">
            <div class="config-label">Retrieval (k)</div>
            <div class="config-value">{config["retriever"]["k"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.caption("Built with LangChain, Chroma & Streamlit")
    st.caption("Powered by OpenAI")


# -------------------------
# MAIN CONTENT - TABS
# -------------------------

st.title("Health Assistant App")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Ask Health Question", "Summarize PDF", "Analyse PDF", "Health Journal"])

# ============================================
# TAB 1: ASK HEALTH QUESTION
# ============================================
with tab1:
    st.header("Ask a Health Question")
    
    question = st.text_input(
        "Question",
        placeholder="Type your health question here...",
        key="health_question",
        label_visibility="collapsed"
    )
    
    if st.button("Submit", type="primary"):
        if question:
            with st.spinner("🔍 Searching for answer..."):
                try:
                    response = chain.invoke(question)
                    answer_text = response.content.strip()
                    
                    if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                        web_results = serpapi_search(question)
                        
                        if web_results:
                            combined_answer = "Here's what I found from web search:\n\n"
                            for result in web_results:
                                combined_answer += f"**{result['title']}**\n{result['snippet']}\n\n"
                            
                            st.session_state.answer = combined_answer
                            st.session_state.sources = [(f"Source {i}", result['url']) for i, result in enumerate(web_results, 1)]
                        else:
                            st.session_state.answer = "I don't have enough information to answer this question."
                            st.session_state.sources = []
                    else:
                        st.session_state.answer = answer_text
                        st.session_state.sources = [("Source", "PDF Knowledge Base")]
                    
                except Exception as e:
                    st.session_state.answer = f"Error: {str(e)}"
                    st.session_state.sources = []
        else:
            st.warning("Please enter a question")
    
    if st.session_state.answer:
        st.subheader("Answer:")
        
        st.markdown(f"""
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 20px;">
            <p>{st.session_state.answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.sources:
            st.write("")
            for source_label, source_url in st.session_state.sources:
                if source_url != "PDF Knowledge Base":
                    st.markdown(f"**{source_label}:** [{source_url}]({source_url})")
                else:
                    st.markdown(f"**{source_label}:** {source_url}")


# ============================================
# TAB 2: SUMMARIZE PDF
# ============================================
with tab2:
    st.header("Summarize PDF")
    
    if config["pdf_files"]:
        selected_pdf_tab = st.selectbox(
            "Select a PDF to summarize",
            options=config["pdf_files"],
            format_func=lambda x: os.path.basename(x),
            key="pdf_tab_select"
        )
        
        if st.button("Generate Summary", type="primary", key="tab_summary_btn"):
            with st.spinner("Generating summary..."):
                summary = summarize_pdf(selected_pdf_tab)
                
                st.subheader("Summary:")
                st.markdown(f"""
                <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 20px;">
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No PDFs available to summarize.")


# ============================================
# TAB 3: ANALYSE PDF (BLOOD WORK/LAB RESULTS)
# ============================================
with tab3:
    st.header("Analyse PDF")
    st.markdown("Upload a PDF of your blood analysis or lab report for AI analysis.")
    
    # File uploader for lab report
    uploaded_lab_pdf = st.file_uploader(
        "Upload Lab Report PDF",
        type=['pdf'],
        key="lab_pdf_upload",
        help="Upload your blood work, lab results, or medical test report"
    )
    
    if uploaded_lab_pdf is not None:
        # Show file info
        st.success(f"✅ Uploaded: {uploaded_lab_pdf.name}")
        
        if st.button("🔬 Analyse Report", type="primary", key="analyse_lab_btn"):
            with st.spinner("Analyzing your lab report..."):
                try:
                    # Extract text from PDF
                    from pypdf import PdfReader
                    from io import BytesIO
                    
                    # Read PDF
                    pdf_file = BytesIO(uploaded_lab_pdf.read())
                    pdf_reader = PdfReader(pdf_file)
                    
                    # Extract text from all pages
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text() + "\n"
                    
                    if not pdf_text.strip():
                        st.error("Could not extract text from PDF. The file may be an image-based PDF.")
                    else:
                        # Create analysis prompt
                        analysis_prompt = f"""You are a medical AI assistant analyzing lab results. 
                        
Please analyze the following lab report and provide:

1. **Key Findings**: List the main test results with their values
2. **Normal vs. Abnormal**: Identify which values are outside normal ranges
3. **Health Implications**: Explain what the results might indicate
4. **Recommendations**: Suggest next steps (e.g., follow-up tests, lifestyle changes, consult doctor)

IMPORTANT: 
- This is for informational purposes only
- Always recommend consulting with a healthcare provider
- Be clear about which values are concerning

Lab Report Content:
{pdf_text[:4000]}

Analysis:"""
                        
                        # Get AI analysis
                        analysis_response = llm.invoke(analysis_prompt)
                        analysis = analysis_response.content
                        
                        # Display results
                        st.subheader("🔬 Analysis Results:")
                        
                        # Use container with custom styling for better rendering
                        with st.container():
                            st.markdown(analysis)
                        
                        st.markdown("---")  # Divider
                        
                        # Medical disclaimer
                        st.warning("""
                        ⚠️ **MEDICAL DISCLAIMER**: This analysis is for informational purposes only and should NOT be considered medical advice. 
                        Always consult with a qualified healthcare provider to interpret your lab results.
                        """)
                        
                except Exception as e:
                    st.error(f"Error analyzing PDF: {str(e)}")
    else:
        st.info("👆 Upload a PDF of your lab report to get started")


# ============================================
# TAB 4: HEALTH JOURNAL (WITH FILE ATTACHMENTS)
# ============================================
with tab4:
    st.header("Health Journal")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        journal_title = st.text_input(
            "Title",
            placeholder="Entry title...",
            key="journal_tab_title"
        )
        
        journal_entry_tab = st.text_area(
            "Journal Entry",
            placeholder="How are you feeling today?",
            height=150,
            key="journal_tab_entry"
        )
        
        # File uploader for attachments (NEW)
        uploaded_file = st.file_uploader(
            "📎 Attach File (Optional)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif'],
            key=f"journal_file_upload_{st.session_state.file_uploader_key}",
            help="Attach lab reports, images, or other documents"
        )
    
    with col2:
        journal_date_tab = st.date_input("Date", key="journal_tab_date")
        
        if st.button("Add Entry", type="primary", use_container_width=True, key="journal_tab_add"):
            if journal_title and journal_entry_tab:
                # Create entry data
                entry_data = {
                    "title": journal_title,
                    "date": journal_date_tab.strftime("%Y-%m-%d"),
                    "entry": journal_entry_tab,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Handle file attachment if present
                if uploaded_file is not None:
                    # Create user's attachment folder
                    attachment_dir = f"journal_attachments/{username}"
                    os.makedirs(attachment_dir, exist_ok=True)
                    
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = uploaded_file.name.split('.')[-1]
                    safe_filename = f"{timestamp}_{uploaded_file.name}"
                    file_path = os.path.join(attachment_dir, safe_filename)
                    
                    # Save file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add file info to entry
                    entry_data["attachment"] = {
                        "filename": uploaded_file.name,
                        "filepath": file_path,
                        "type": file_extension.lower()
                    }
                
                st.session_state.journal_entries.append(entry_data)
                save_user_data()  # Save to file
                
                # Clear file uploader by changing its key
                st.session_state.file_uploader_key += 1
                
                # Clear text input fields
                st.session_state.journal_tab_title = ""
                st.session_state.journal_tab_entry = ""
                
                st.success("✅ Entry added!")
                st.rerun()
            else:
                st.warning("Please enter both title and entry text")
    
    st.divider()
    
    if st.session_state.journal_entries:
        st.subheader("Journal Entries")
        
        for entry in reversed(st.session_state.journal_entries):
            # Show attachment indicator in header if present
            attachment_indicator = " 📎" if "attachment" in entry else ""
            
            with st.expander(f"📅 {entry['date']} - {entry.get('title', 'Untitled')}{attachment_indicator}"):
                st.write(entry['entry'])
                
                # Display attachment if present
                if "attachment" in entry:
                    st.divider()
                    attachment = entry["attachment"]
                    file_type = attachment["type"]
                    
                    # Display based on file type
                    if file_type in ['png', 'jpg', 'jpeg', 'gif']:
                        # Show image
                        st.image(attachment["filepath"], caption=attachment["filename"], use_container_width=True)
                    elif file_type == 'pdf':
                        # Show PDF info with download button
                        st.info(f"📄 PDF: {attachment['filename']}")
                        
                        # Read and provide download button
                        with open(attachment["filepath"], "rb") as f:
                            st.download_button(
                                label="📥 Download PDF",
                                data=f,
                                file_name=attachment["filename"],
                                mime="application/pdf",
                                key=f"download_{entry['timestamp']}"
                            )
                
                # Delete button
                if st.button("🗑️ Delete", key=f"delete_tab_journal_{entry['timestamp']}"):
                    # Try to delete attachment file if exists
                    file_deleted = True
                    if "attachment" in entry and "filepath" in entry["attachment"]:
                        filepath = os.path.normpath(entry["attachment"]["filepath"])
                        
                        if os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                            except Exception as e:
                                file_deleted = False
                                st.warning(f"⚠️ Could not delete file: {str(e)}")
                    
                    # Remove entry from list
                    for idx, e in enumerate(st.session_state.journal_entries):
                        if e['timestamp'] == entry['timestamp']:
                            st.session_state.journal_entries.pop(idx)
                            break
                    
                    save_user_data()
                    
                    if file_deleted:
                        st.success("✅ Entry and attachment deleted!")
                    else:
                        st.info("✅ Entry deleted (attachment file could not be removed)")
                    
                    st.rerun()
    else:
        st.info("No journal entries yet.")
