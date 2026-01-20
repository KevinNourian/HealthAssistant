# main.py

"""
Health Assistant - Streamlit App
RAG-based Q&A system with web fallback using Chroma vector store.

Features:
- Export Chat History
- Health Reminders
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
    layout="centered",
    initial_sidebar_state="expanded"
)


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
    """
    Initialize vector store - runs once and caches the result.
    This prevents reloading on every Streamlit interaction.
    """
    vectorstore = get_or_create_vectorstore(
        pdf_paths=config["pdf_files"],
        persist_directory=config["chroma_directory"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        force_recreate=False
    )
    return get_retriever(vectorstore, k=config["retriever"]["k"])


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
    retriever = initialize_vectorstore()
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
# Web Search Fallback Function
# -------------------------
def serpapi_search(query: str, max_results: int = 3) -> tuple:
    """
    Search the web using SerpAPI when PDF doesn't have the answer.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Tuple of (formatted_results, source_urls)
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    
    try:
        search = GoogleSearch(params)
        result = search.get_dict()
        answers = []
        urls = []
        
        if "organic_results" in result:
            for item in result["organic_results"][:max_results]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if title and snippet:
                    answers.append(f"**{title}**\n{snippet}")
                    urls.append(link)
        
        formatted_answer = "\n\n".join(answers) if answers else "No results found."
        return formatted_answer, urls
    
    except Exception as e:
        return f"Search error: {str(e)}", []


# -------------------------
# Initialize Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "reminders" not in st.session_state:
    st.session_state.reminders = []


# -------------------------
# Streamlit UI - Main App
# -------------------------

# Title and description
st.title("🏥 Health Assistant")
st.caption("💬 Ask questions about health topics from your knowledge base")

# Disclaimer
with st.expander("⚠️ Important Disclaimer - Please Read"):
    st.warning("""
    **This information is for educational purposes only.**
    
    - This AI assistant provides information based on uploaded PDF documents
    - It is NOT a substitute for professional medical advice, diagnosis, or treatment
    - Always consult qualified healthcare professionals for medical concerns
    - In case of emergency, call your local emergency services
    """)


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            if "urls" in message and message["urls"]:
                st.caption("*Sources:*")
                for i, url in enumerate(message["urls"], 1):
                    st.caption(f"{i}. {url}")
            else:
                st.caption(f"*Source: {message['source']}*")


# Chat input
if prompt := st.chat_input("Ask a health question... (e.g., 'What are COVID-19 symptoms?')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                # Query RAG system
                response = chain.invoke(prompt)
                answer_text = response.content.strip()
                
                # Check if answer was found in PDFs
                if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                    st.info("📡 No answer found in PDFs, searching the web...")
                    answer_text, source_urls = serpapi_search(prompt)
                    source = "Web Search"
                    
                    # Display answer
                    st.markdown(answer_text)
                    
                    # Display source URLs
                    if source_urls:
                        st.caption("*Sources:*")
                        for i, url in enumerate(source_urls, 1):
                            st.caption(f"{i}. {url}")
                    else:
                        st.caption(f"*Source: {source}*")
                else:
                    source = "PDF Knowledge Base"
                    
                    # Display answer
                    st.markdown(answer_text)
                    st.caption(f"*Source: {source}*")
                
                # Add to chat history
                message_data = {
                    "role": "assistant",
                    "content": answer_text,
                    "source": source
                }
                
                # Add URLs if from web search
                if source == "Web Search" and source_urls:
                    message_data["urls"] = source_urls
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "source": "Error"
                })


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    # ============================================
    # FEATURE: EXPORT CHAT HISTORY
    # ============================================
    st.header("💾 Export Chat")
    
    if st.session_state.messages:
        # Generate chat transcript
        chat_text = f"Health Assistant Chat History\n"
        chat_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_text += "="*50 + "\n\n"
        
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            source = msg.get("source", "")
            urls = msg.get("urls", [])
            
            chat_text += f"{role}:\n{content}\n"
            if source:
                chat_text += f"[Source: {source}]\n"
            if urls:
                chat_text += "Sources:\n"
                for i, url in enumerate(urls, 1):
                    chat_text += f"{i}. {url}\n"
            chat_text += "\n" + "-"*50 + "\n\n"
        
        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name=f"health_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("No chat history to export yet")
    
    st.divider()
    
    # ============================================
    # FEATURE: HEALTH REMINDERS
    # ============================================
    st.header("⏰ Health Reminders")
    
    # Add new reminder
    with st.form("reminder_form"):
        reminder_text = st.text_input("Reminder", placeholder="Doctor visit")
        reminder_date = st.date_input("Date")
        
        col1, col2 = st.columns(2)
        with col1:
            add_reminder = st.form_submit_button("➕ Add", use_container_width=True)
        with col2:
            clear_all = st.form_submit_button("🗑️ Clear All", use_container_width=True)
        
        if add_reminder and reminder_text:
            st.session_state.reminders.append({
                "text": reminder_text,
                "date": reminder_date.strftime("%Y-%m-%d"),
                "id": len(st.session_state.reminders)
            })
            st.success("✅ Reminder added!")
            st.rerun()
        
        if clear_all:
            st.session_state.reminders = []
            st.success("✅ All reminders cleared!")
            st.rerun()
    
    # Display reminders
    if st.session_state.reminders:
        st.subheader("Active Reminders:")
        for i, reminder in enumerate(st.session_state.reminders):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📅 **{reminder['date']}** - {reminder['text']}")
            with col2:
                if st.button("❌", key=f"delete_{i}"):
                    st.session_state.reminders.pop(i)
                    st.rerun()
    else:
        st.info("No active reminders")
    
    st.divider()
    
    # ============================================
    # KNOWLEDGE BASE INFO
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
    # CONFIGURATION INFO
    # ============================================
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", config["llm"]["model"])
        st.metric("Chunk Size", config["chunking"]["chunk_size"])
    with col2:
        st.metric("Temperature", config["llm"]["temperature"])
        st.metric("Retrieval (k)", config["retriever"]["k"])
    
    st.divider()
    
    # ============================================
    # ACTIONS
    # ============================================
    st.header("🛠️ Actions")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Show example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the symptoms of COVID-19?
        - How is diabetes managed?
        - What are risk factors for heart disease?
        - Tell me about vaccine information
        - What medications treat high blood pressure?
        """)
    
    st.divider()
    
    # Footer
    st.caption("Built with LangChain, Chroma & Streamlit")
    st.caption("Powered by OpenAI")
