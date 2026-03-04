<p align="center">
  <img src="assets/banner.svg" alt="Health Assistant Banner" width="100%" />
</p>

<p align="center">
  <strong>A multi-user, RAG-powered health assistant with safety guardrails, built on Streamlit.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangChain-0.1+-1C3C3C?logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-0.4+-00A67E" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI" />
</p>

---

## Overview

Health Assistant is a Retrieval-Augmented Generation (RAG) application that answers health-related questions by combining a personal PDF knowledge base with web search. It features multi-user authentication, crisis detection guardrails, rate limiting, a health journal, and document management &mdash; all through a clean Streamlit web interface.

## Features

- **RAG-Powered Q&A** &mdash; Queries a vector store of health PDFs using hybrid retrieval (BM25 + semantic search with Reciprocal Rank Fusion), with automatic web fallback via SerpAPI
- **Multi-User Authentication** &mdash; Cookie-based session management with per-user data isolation for reminders, journals, and chat history
- **Safety Guardrails** &mdash; Crisis detection (regex patterns for medical/mental health emergencies) and LLM-based health topic validation to keep conversations on-topic
- **Rate Limiting** &mdash; Per-user sliding-window rate limiter (configurable, default 20 requests/hour)
- **Document Management** &mdash; Upload and remove PDFs through the UI with automatic vector store sync
- **Health Journal** &mdash; Track health entries with optional file attachments (PDFs, images)
- **Reminders** &mdash; Date-based health reminders displayed in the sidebar
- **Tool-Calling Agent** &mdash; Four specialized tools: knowledge base search, web search, document summarization, and lab report analysis

## Project Structure

```
Health_Assistant/
├── app.py                     # Main Streamlit application
├── config.json                # Runtime configuration
├── credentials.yaml           # Authentication credentials
├── pyproject.toml             # Project metadata & dependencies
│
├── core/                      # Application modules
│   ├── agent.py               # Agent loop with tool binding
│   ├── auth.py                # Authentication setup
│   ├── config.py              # Config loading / saving
│   ├── guardrails.py          # Crisis detection & topic validation
│   ├── prompts.py             # System & tool prompt templates
│   ├── rate_limiter.py        # Per-user rate limiter
│   ├── tools.py               # LangChain tool definitions
│   ├── user_data.py           # User data persistence
│   └── vector_store.py        # Chroma vector store & hybrid retriever
│
├── scripts/                   # Utilities
│   ├── rebuild_vectorstore.py # Force-rebuild the vector store
│   └── generate_credentials.py
│
├── data/                      # Knowledge base PDFs & vector DB
│   ├── COVID-19.pdf
│   ├── Diabetes.pdf
│   ├── Hypertension.pdf
│   ├── Heart_Disease.pdf
│   ├── Physical_wellness.pdf
│   ├── Cancer.pdf
│   ├── Mental_Disorders.pdf
│   └── chroma_db/             # Persisted Chroma database
│
├── user_data/                 # Per-user JSON data files
├── journal_attachments/       # Journal file uploads (per user)
└── assets/                    # Static assets (CSS, images)
    ├── banner.svg
    └── style.css
```

## Quick Start

### 1. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

### 3. Run the Application

```bash
streamlit run app.py
```

On first launch, the vector store will be built from the PDFs in `data/` (this may take a few minutes). Subsequent launches load the persisted database instantly.

### 4. Log In

Use one of the demo accounts or create your own via `scripts/generate_credentials.py`:

| Username | Password |
|----------|----------|
| alice    | temp123  |
| bob      | temp456  |

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "pdf_files": ["data/COVID-19.pdf", "data/Diabetes.pdf", "..."],
  "chroma_directory": "data/chroma_db",
  "chunking": {
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "retriever": { "k": 3 },
  "llm": {
    "model": "gpt-4o-mini",
    "temperature": 0
  },
  "rate_limit": {
    "max_requests": 20,
    "window_seconds": 3600
  }
}
```

| Option | Description |
|--------|-------------|
| `pdf_files` | PDFs included in the knowledge base |
| `chroma_directory` | Vector database storage path |
| `chunk_size` | Characters per document chunk |
| `chunk_overlap` | Overlap between chunks for context continuity |
| `retriever.k` | Number of chunks retrieved per query |
| `llm.model` | OpenAI model (e.g. `gpt-4o-mini`, `gpt-4`) |
| `llm.temperature` | 0 = deterministic, 1 = creative |
| `rate_limit.max_requests` | Max requests per user per window |
| `rate_limit.window_seconds` | Sliding window duration in seconds |

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  Guardrails │──→ Crisis? → Emergency resources
│  (Safety)   │──→ Off-topic? → Redirect
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Rate Limiter│──→ Over limit? → Wait message
└─────┬───────┘
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│   Agent     │────→│  Tool: Search KB │──→ Hybrid Retriever (BM25 + Semantic)
│  (LLM Loop) │────→│  Tool: Web Search│──→ SerpAPI
│             │────→│  Tool: Summarize │──→ Document summary
│             │────→│  Tool: Lab Report│──→ Educational analysis
└─────┬───────┘     └──────────────────┘
      │
      ▼
   Response with sources
```

## Safety

Health Assistant includes multiple safety layers:

- **Crisis Detection** &mdash; Regex patterns identify medical emergencies (chest pain, stroke symptoms) and mental health crises (suicidal ideation, self-harm), immediately providing emergency resources (911, 988 Suicide & Crisis Lifeline)
- **Health Topic Validation** &mdash; An LLM classifier ensures queries are health-related before processing
- **Content Boundaries** &mdash; The system prompt prohibits diagnosis, medication prescriptions, and treatment plans; all responses include a disclaimer to consult a healthcare professional
- **Rate Limiting** &mdash; Prevents abuse with per-user request throttling

## Updating the Knowledge Base

### Via the UI

Navigate to the **Manage Documents** tab to upload or remove PDFs. The vector store updates automatically.

### Manually

1. Place new PDFs in `data/`
2. Add their paths to `config.json`
3. Rebuild:
   ```bash
   python scripts/rebuild_vectorstore.py
   ```

## Built With

- [Streamlit](https://streamlit.io/) &mdash; Web interface
- [LangChain](https://www.langchain.com/) &mdash; Agent framework & tool orchestration
- [ChromaDB](https://www.trychroma.com/) &mdash; Vector storage & semantic search
- [OpenAI](https://openai.com/) &mdash; LLM (GPT-4o-mini) & embeddings
- [SerpAPI](https://serpapi.com/) &mdash; Web search fallback


