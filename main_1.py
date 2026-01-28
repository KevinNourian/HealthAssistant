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
    /* ═══════════════════════════════════════════════════════════════════════
       MINIMALIST NORDIC THEME
       - Pure whites & soft grays
       - Coral accent (#FF6B6B)
       - Maximum whitespace
       - Clean typography (Inter)
       - Subtle interactions
    ═══════════════════════════════════════════════════════════════════════ */
    
    /* Import Inter font - clean, modern, highly readable */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ─────────────────────────────────────────────────────────────────────
       CSS Variables for easy theming
    ───────────────────────────────────────────────────────────────────── */
    :root {
        --white: #FFFFFF;
        --off-white: #FAFAFA;
        --light-gray: #F5F5F5;
        --border-gray: #E8E8E8;
        --medium-gray: #D0D0D0;
        --text-gray: #6B7280;
        --text-dark: #1F2937;
        --text-black: #111827;
        --accent: #FF6B6B;
        --accent-hover: #FF5252;
        --accent-light: #FFF0F0;
        --success: #10B981;
        --success-light: #D1FAE5;
        --warning: #F59E0B;
        --warning-light: #FEF3C7;
        --error: #EF4444;
        --error-light: #FEE2E2;
        --info: #6B7280;
        --info-light: #F3F4F6;
        --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.04);
        --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.06);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --transition: all 0.2s ease;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Global Styles
    ───────────────────────────────────────────────────────────────────── */
    /* Apply font to main content areas only */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp div, .stApp label, 
    .stApp input, .stApp textarea,
    .stApp a, .stApp li, .stApp td, .stApp th,
    .stMarkdown, .stText {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background-color: var(--off-white);
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Typography
    ───────────────────────────────────────────────────────────────────── */
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: var(--text-black) !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 0.25rem !important;
    }
    
    h2 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-dark) !important;
        letter-spacing: -0.3px !important;
        margin-top: 0 !important;
    }
    
    h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text-dark) !important;
    }
    
    .stMarkdown p, .stMarkdown div, .stMarkdown label {
        color: var(--text-dark) !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Sidebar - Clean white with subtle border
    ───────────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--white) !important;
        border-right: 1px solid var(--border-gray) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 1rem 0.5rem !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--text-dark) !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-gray) !important;
        font-size: 0.875rem !important;
    }
    
    /* Sidebar buttons - pure white to match Browse Files */
    [data-testid="stSidebar"] .stButton > button {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1rem !important;
        transition: var(--transition) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Delete reminder button - pure white with bigger X */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.625rem !important;
        min-width: 32px !important;
        min-height: 32px !important;
        line-height: 1 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Tabs - Minimal underline style
    ───────────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-gray) !important;
        padding: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        padding: 0 24px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: var(--text-gray) !important;
        transition: var(--transition) !important;
        white-space: nowrap !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-dark) !important;
        background: transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-dark) !important;
        border-bottom: 2px solid var(--text-dark) !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Buttons - Pure white to match Browse Files button
       Use specific selectors to avoid affecting sidebar collapse button
    ───────────────────────────────────────────────────────────────────── */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.625rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: var(--transition) !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-subtle) !important;
    }
    
    .stButton > button:active,
    [data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary/outline buttons */
    .stButton > button[kind="secondary"] {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
        box-shadow: none !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Form Inputs - Clean, minimal borders
    ───────────────────────────────────────────────────────────────────── */
    .stTextInput input,
    .stTextArea textarea,
    .stDateInput input {
        background-color: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.625rem 0.875rem !important;
        font-size: 0.875rem !important;
        color: var(--text-dark) !important;
        transition: var(--transition) !important;
    }
    
    /* Selectbox - ensure selected value is visible */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
    }
    
    /* Selectbox value text */
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"],
    .stSelectbox [data-baseweb="select"] span {
        color: var(--text-dark) !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stSelectbox [data-baseweb="select"]:focus-within,
    .stDateInput input:focus {
        border-color: var(--medium-gray) !important;
        box-shadow: 0 0 0 3px var(--light-gray) !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--medium-gray) !important;
    }
    
    /* Labels */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stDateInput label {
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        color: var(--text-gray) !important;
        margin-bottom: 0.375rem !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       File Uploader - Minimal dashed border
    ───────────────────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--white) !important;
        border: 1px dashed var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.5rem !important;
        transition: var(--transition) !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--medium-gray) !important;
        background: var(--off-white) !important;
    }
    
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        padding: 0 !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Alert Messages - Subtle left border style
    ───────────────────────────────────────────────────────────────────── */
    .stSuccess {
        background-color: var(--success-light) !important;
        border: none !important;
        border-left: 3px solid var(--success) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1rem !important;
    }
    
    .stSuccess p {
        color: #065F46 !important;
        font-size: 0.875rem !important;
    }
    
    .stInfo {
        background-color: var(--info-light) !important;
        border: none !important;
        border-left: 3px solid var(--info) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1rem !important;
    }
    
    .stInfo p {
        color: var(--text-dark) !important;
        font-size: 0.875rem !important;
    }
    
    .stWarning {
        background-color: var(--warning-light) !important;
        border: none !important;
        border-left: 3px solid var(--warning) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1rem !important;
    }
    
    .stWarning p {
        color: #92400E !important;
        font-size: 0.875rem !important;
    }
    
    .stError {
        background-color: var(--error-light) !important;
        border: none !important;
        border-left: 3px solid var(--error) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.875rem 1rem !important;
    }
    
    .stError p {
        color: #991B1B !important;
        font-size: 0.875rem !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Expanders - Clean minimal style (if used elsewhere)
    ───────────────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-md) !important;
        background: var(--white) !important;
    }
    
    [data-testid="stExpander"] summary {
        background: var(--white) !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        color: var(--text-dark) !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Sidebar collapse button - let Streamlit handle this natively
    ───────────────────────────────────────────────────────────────────── */
    /* No custom styles - preserve default Streamlit behavior */
    
    /* ─────────────────────────────────────────────────────────────────────
       Dividers - Subtle and light
    ───────────────────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background-color: var(--border-gray) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Cards - Answer/Summary display boxes (full width)
    ───────────────────────────────────────────────────────────────────── */
    .answer-card {
        background: var(--white);
        border: 1px solid var(--border-gray);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: var(--shadow-subtle);
        width: 100%;
        max-width: 100%;
    }
    
    .answer-card p {
        color: var(--text-dark);
        font-size: 0.9375rem;
        line-height: 1.7;
        margin: 0;
    }
    
    /* st.container with border styling */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--white) !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--radius-lg) !important;
        padding: 0 !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1rem 1.25rem !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Welcome header styling
    ───────────────────────────────────────────────────────────────────── */
    .welcome-header {
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--text-dark);
        margin-bottom: 0.5rem;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Config metrics - sidebar display
    ───────────────────────────────────────────────────────────────────── */
    .config-item {
        margin-bottom: 0.75rem;
    }
    
    .config-label {
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-gray);
        margin-bottom: 0.125rem;
    }
    
    .config-value {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-dark);
        word-break: break-word;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Reminder items
    ───────────────────────────────────────────────────────────────────── */
    .reminder-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.625rem 0;
        border-bottom: 1px solid var(--light-gray);
    }
    
    .reminder-item:last-child {
        border-bottom: none;
    }
    
    .reminder-date {
        font-size: 0.75rem;
        color: var(--text-gray);
        font-weight: 500;
    }
    
    .reminder-text {
        font-size: 0.8125rem;
        color: var(--text-dark);
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Medical disclaimer box
    ───────────────────────────────────────────────────────────────────── */
    .disclaimer-box {
        background: var(--warning-light);
        border-left: 3px solid var(--warning);
        border-radius: var(--radius-sm);
        padding: 1rem;
        margin-top: 1.5rem;
    }
    
    .disclaimer-box p {
        color: #92400E;
        font-size: 0.8125rem;
        margin: 0;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Hide Streamlit branding & form hints
    ───────────────────────────────────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide "Press Enter to submit form" and ALL form hints comprehensively */
    .stForm [data-testid="stFormSubmitContent"],
    [data-testid="stFormSubmitButton"] ~ div,
    .stFormSubmitContent,
    [data-testid="stForm"] small,
    .stTextArea [data-testid="stMarkdownContainer"] small,
    .stTextArea small,
    [data-testid="stForm"] [data-testid="InputInstructions"],
    [data-testid="InputInstructions"],
    .stForm div[data-testid="InputInstructions"],
    div[data-baseweb="form-control-counter"],
    .stTextInput + div small,
    .stForm .stTextInput + div,
    [data-testid="stForm"] > div > div > div > small,
    [data-testid="stForm"] span:has(+ button) small,
    /* Target all small/caption text in forms */
    [data-testid="stForm"] .stCaption,
    [data-testid="stForm"] [data-testid="stCaptionContainer"],
    form small,
    form .stCaption,
    /* Streamlit form instructions wrapper */
    [data-testid="stForm"] > div:last-child > div:last-child > small,
    [data-testid="stForm"] > div > div:has(button) + div,
    /* Hide any remaining instruction text */
    .stForm > div > div > div:has(small):not(:has(input)):not(:has(button)) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Download button styling - pure white
    ───────────────────────────────────────────────────────────────────── */
    .stDownloadButton button {
        background-color: var(--white) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-gray) !important;
        font-size: 0.8125rem !important;
    }
    
    .stDownloadButton button:hover {
        background-color: var(--light-gray) !important;
        border-color: var(--medium-gray) !important;
    }
    
    /* ─────────────────────────────────────────────────────────────────────
       Scrollbar styling
    ───────────────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-gray);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--medium-gray);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-gray);
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
    st.session_state.answer = ""
    st.session_state.sources = []
    st.session_state.file_uploader_key = 0
    st.session_state.pdf_summary = ""
    st.session_state.lab_analysis = ""
    st.session_state.lab_error = None

if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""
if "lab_analysis" not in st.session_state:
    st.session_state.lab_analysis = ""
if "lab_error" not in st.session_state:
    st.session_state.lab_error = None


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
    st.markdown(f"<p class='welcome-header'>Welcome, {name}</p>", unsafe_allow_html=True)
    
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
# TAB 1: ASK HEALTH QUESTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a Health Question")
    st.markdown("<p style='color: #6B7280; font-size: 14px; margin-bottom: 1.5rem;'>Get answers from your knowledge base or the web</p>", unsafe_allow_html=True)
    
    question = st.text_input(
        "Question",
        placeholder="Type your health question here...",
        key="health_question",
        label_visibility="collapsed"
    )
    
    if st.button("Ask", type="primary", key="ask_btn"):
        if question:
            with st.spinner("Searching..."):
                try:
                    response = chain.invoke(question)
                    answer_text = response.content.strip()
                    
                    if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                        web_results = serpapi_search(question)
                        
                        if web_results:
                            combined_answer = "Here's what I found:\n\n"
                            for result in web_results:
                                combined_answer += f"**{result['title']}**\n{result['snippet']}\n\n"
                            
                            st.session_state.answer = combined_answer
                            st.session_state.sources = [
                                (f"Source {i}", result['url']) 
                                for i, result in enumerate(web_results, 1)
                            ]
                        else:
                            st.session_state.answer = "I couldn't find enough information to answer this question."
                            st.session_state.sources = []
                    else:
                        st.session_state.answer = answer_text
                        st.session_state.sources = [("Source", "Knowledge Base")]
                    
                except Exception as e:
                    st.session_state.answer = f"Error: {str(e)}"
                    st.session_state.sources = []
        else:
            st.warning("Please enter a question")
    
    # Display answer (full width with proper markdown rendering)
    if st.session_state.answer:
        st.markdown("#### Answer")
        with st.container(border=True):
            st.markdown(st.session_state.answer)
        
        if st.session_state.sources:
            st.markdown("")
            for source_label, source_url in st.session_state.sources:
                if source_url != "Knowledge Base":
                    st.markdown(f"<p style='font-size: 13px; color: #6B7280;'>{source_label}: <a href='{source_url}' style='color: #1F2937;'>{source_url[:60]}...</a></p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size: 13px; color: #6B7280;'>{source_label}: {source_url}</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SUMMARIZE PDF
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Summarize Document")
    st.markdown("<p style='color: #6B7280; font-size: 14px; margin-bottom: 1.5rem;'>Generate AI summaries of your health documents</p>", unsafe_allow_html=True)
    
    if config["pdf_files"]:
        selected_pdf = st.selectbox(
            "Select document",
            options=config["pdf_files"],
            format_func=lambda x: os.path.basename(x),
            key="pdf_select",
            label_visibility="collapsed"
        )
        
        if st.button("Summarize", type="primary", key="summarize_btn"):
            with st.spinner("Generating summary..."):
                summary = summarize_pdf(selected_pdf)
                st.session_state.pdf_summary = summary
        
        # Display summary if exists (full width with proper markdown rendering)
        if "pdf_summary" in st.session_state and st.session_state.pdf_summary:
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
    st.markdown("<p style='color: #6B7280; font-size: 14px; margin-bottom: 1.5rem;'>Upload your blood work or lab results for AI analysis</p>", unsafe_allow_html=True)
    
    uploaded_lab_pdf = st.file_uploader(
        "Upload Lab Report",
        type=['pdf'],
        key="lab_pdf_upload",
        help="Upload your blood work, lab results, or medical test report",
        label_visibility="collapsed"
    )
    
    if uploaded_lab_pdf is not None:
        st.success(f"Uploaded: {uploaded_lab_pdf.name}")
        
        if st.button("Analyse", type="primary", key="analyse_btn"):
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
        
        # Display analysis results (full width)
        if "lab_error" in st.session_state and st.session_state.lab_error:
            st.error(st.session_state.lab_error)
        
        if "lab_analysis" in st.session_state and st.session_state.lab_analysis:
            st.markdown("#### Analysis Results")
            # Use st.container with st.markdown for proper markdown rendering
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
    st.markdown("<p style='color: #6B7280; font-size: 14px; margin-bottom: 1.5rem;'>Track your health journey with notes and attachments</p>", unsafe_allow_html=True)
    
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
            key=f"journal_file_{st.session_state.file_uploader_key}",
            label_visibility="collapsed"
        )
    
    with col2:
        journal_date = st.date_input("Date", key="journal_date")
        
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        
        if st.button("Save Entry", type="primary", use_container_width=True):
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
                st.success("Entry saved!")
                st.rerun()
            else:
                st.warning("Please enter both title and entry")
    
    st.markdown("---")
    
    if st.session_state.journal_entries:
        st.markdown("#### Past Entries")
        
        for entry in reversed(st.session_state.journal_entries):
            attachment_icon = " 📎" if "attachment" in entry else ""
            
            # Use container with border instead of expander for better rendering
            with st.container(border=True):
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
                
                if st.button("Delete Entry", key=f"del_{entry['timestamp']}"):
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
