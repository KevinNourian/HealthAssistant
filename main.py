# main.py

"""
Health Assistant - Streamlit App
Tabbed main interface with streamlined sidebar
"""

import streamlit as st
from dotenv import load_dotenv
import os
import json
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
# Initialize Session State
# -------------------------
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = []


# -------------------------
# SIDEBAR (STREAMLINED)
# -------------------------
with st.sidebar:
    # ============================================
    # HEALTH REMINDERS (NO CLEAR ALL BUTTON)
    # ============================================
    st.header("⏰ Health Reminders")
    
    with st.form("reminder_form", clear_on_submit=True):
        reminder_text = st.text_input("Reminder", placeholder="Doctor visit", key="reminder_input")
        reminder_date = st.date_input("Date", key="reminder_date")
        
        # Only Add button (no Clear All)
        add_reminder = st.form_submit_button("➕ Add", use_container_width=True)
        
        if add_reminder and reminder_text:
            st.session_state.reminders.append({
                "text": reminder_text,
                "date": reminder_date.strftime("%Y-%m-%d"),
                "id": len(st.session_state.reminders)
            })
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
                    st.rerun()
    else:
        st.info("No active reminders")
    
    st.divider()
    
    # ============================================
    # KNOWLEDGE BASE (NO EXAMPLE QUESTIONS)
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
# MAIN CONTENT - BIGGER, BOLDER TABS
# -------------------------

st.title("Health Assistant App")

# Create tabs with custom styling (made bigger and bolder via CSS above)
tab1, tab2, tab3 = st.tabs(["Ask Health Question", "Summarize PDF", "Health Journal"])

# ============================================
# TAB 1: ASK HEALTH QUESTION
# ============================================
with tab1:
    st.header("Ask a Health Question")
    
    # Question input
    question = st.text_input(
        "Question",
        placeholder="Type your health question here...",
        key="health_question",
        label_visibility="collapsed"
    )
    
    # Submit button
    if st.button("Submit", type="primary"):
        if question:
            with st.spinner("🔍 Searching for answer..."):
                try:
                    # Query RAG system
                    response = chain.invoke(question)
                    answer_text = response.content.strip()
                    
                    # Check if answer was found in PDFs
                    if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                        # Search web
                        web_results = serpapi_search(question)
                        
                        if web_results:
                            # Combine web results
                            combined_answer = "Here's what I found from web search:\n\n"
                            for result in web_results:
                                combined_answer += f"**{result['title']}**\n{result['snippet']}\n\n"
                            
                            st.session_state.answer = combined_answer
                            st.session_state.sources = [(f"Source {i}", result['url']) for i, result in enumerate(web_results, 1)]
                        else:
                            st.session_state.answer = "I don't have enough information to answer this question."
                            st.session_state.sources = []
                    else:
                        # PDF answer
                        st.session_state.answer = answer_text
                        st.session_state.sources = [("Source", "PDF Knowledge Base")]
                    
                except Exception as e:
                    st.session_state.answer = f"Error: {str(e)}"
                    st.session_state.sources = []
        else:
            st.warning("Please enter a question")
    
    # Display answer
    if st.session_state.answer:
        st.subheader("Answer:")
        
        # Answer box
        st.markdown(f"""
        <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 20px;">
            <p>{st.session_state.answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources
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
# TAB 3: HEALTH JOURNAL (WITH TITLE FIELD)
# ============================================
with tab3:
    st.header("Health Journal")
    
    # Add entry form with Title field
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Title field (NEW)
        journal_title = st.text_input(
            "Title",
            placeholder="Entry title...",
            key="journal_tab_title"
        )
        
        # Entry field
        journal_entry_tab = st.text_area(
            "Journal Entry",
            placeholder="How are you feeling today?",
            height=150,
            key="journal_tab_entry"
        )
    
    with col2:
        journal_date_tab = st.date_input("Date", key="journal_tab_date")
        
        if st.button("Add Entry", type="primary", use_container_width=True, key="journal_tab_add"):
            if journal_title and journal_entry_tab:
                st.session_state.journal_entries.append({
                    "title": journal_title,
                    "date": journal_date_tab.strftime("%Y-%m-%d"),
                    "entry": journal_entry_tab,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("✅ Entry added!")
                st.rerun()
            else:
                st.warning("Please enter both title and entry text")
    
    st.divider()
    
    # Display entries with Title and Date
    if st.session_state.journal_entries:
        st.subheader("Journal Entries")
        
        for entry in reversed(st.session_state.journal_entries):
            # Show Title and Date in expander header
            with st.expander(f"📅 {entry['date']} - {entry.get('title', 'Untitled')}"):
                st.write(entry['entry'])
                
                if st.button("🗑️ Delete", key=f"delete_tab_journal_{entry['timestamp']}"):
                    for idx, e in enumerate(st.session_state.journal_entries):
                        if e['timestamp'] == entry['timestamp']:
                            st.session_state.journal_entries.pop(idx)
                            break
                    st.rerun()
    else:
        st.info("No journal entries yet.")
