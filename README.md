<p align="center">
  <img src="assets/banner.svg" alt="Health Assistant Banner" width="100%" />
</p>

<p align="center">
  <strong>A multi-user health assistant with agentic RAG, persistent memory, and safety guardrails — built on Streamlit and LangGraph.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangGraph-1.0+-1C3C3C" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-0.1+-1C3C3C?logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-0.4+-00A67E" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/LangSmith-Tracing-FF6B35" alt="LangSmith" />
</p>

---

## Overview

Health Assistant is an agentic RAG application that answers health-related questions by combining a personal PDF knowledge base with web search. It uses a LangGraph StateGraph to orchestrate tool calling, retrieval grading, and persistent memory across sessions. The application features multi-user authentication, crisis detection guardrails, health record tracking, token usage monitoring, and document management — all through a clean Nordic-styled Streamlit web interface.

## Features

### Agent & RAG

- **LangGraph Agent** &mdash; A StateGraph-based ReAct agent that decides which tools to call and when, with conditional routing and a configurable context window
- **Agentic RAG** &mdash; A retrieval grading node evaluates knowledge base results for relevance before the agent uses them; irrelevant results trigger automatic fallback to web search
- **Persistent Memory** &mdash; Short-term memory (conversation history within a session) and long-term memory (SQLite-backed checkpointer that persists across sessions, per user)
- **RAG-Powered Q&A** &mdash; Queries a vector store of health PDFs using hybrid retrieval (BM25 + semantic search with Reciprocal Rank Fusion), with automatic web fallback via SerpAPI
- **Five Agent Tools** &mdash; Knowledge base search, web search, document summarization, lab report analysis, and blood pressure data retrieval
- **Context Window Management** &mdash; Sliding window limits the number of messages sent to the LLM, keeping token costs under control while the full history remains in the checkpoint

### Health Records

- **Blood Pressure Tracker** &mdash; Record systolic, diastolic, and pulse readings; view trends on an interactive Plotly bar chart; ask the agent to analyze your data over time
- **Health Journal** &mdash; Track health entries with optional file attachments (PDFs, images) and markdown formatting support
- **Medical Visit Tracker** &mdash; Log appointments with date, time, doctor name, and markdown notes
- **Vaccination Records** &mdash; Track vaccination history with dates, vaccine names, and notes
- **Prescription Manager** &mdash; Dynamic drug and dose builder with dosage frequency tracking
- **Lab Test Records** &mdash; Record lab results with test type, result values, normal range status, and file upload for report PDFs
- **Reminders** &mdash; Date-based health reminders displayed in the sidebar

### Infrastructure

- **Token Usage & Cost Tracking** &mdash; Per-turn token counts displayed in the chat area; cumulative session totals and cost estimate in the sidebar
- **Multi-User Authentication** &mdash; Cookie-based session management with self-service registration and per-user data isolation
- **Safety Guardrails** &mdash; Crisis detection (regex patterns for medical/mental health emergencies, including third-person phrasing) and LLM-based health topic validation with conversation context awareness
- **Rate Limiting** &mdash; Per-user sliding-window rate limiter with SQLite persistence (configurable, default 20 requests/hour)
- **Document Management** &mdash; Upload and remove PDFs through the UI with automatic vector store sync; document summaries include a transparency note when based on a subset of content
- **Reliability** &mdash; Tenacity retry with exponential backoff on SerpAPI calls; `request_timeout=30` on all LLM calls; graceful error handling in the retrieval grading node so a grading failure never crashes a conversation turn
- **Observability** &mdash; Structured logging throughout; optional LangSmith tracing via environment variables (zero code changes required); BM25 index build time logged at startup

## Project Structure

```
Health_Assistant/
├── app.py                     # Main Streamlit entry point
├── config.json                # Runtime configuration
├── credentials.yaml           # Authentication credentials
├── pyproject.toml             # Project metadata & dependencies
│
├── core/                      # Backend modules
│   ├── agent.py               # LangGraph StateGraph with agentic RAG
│   ├── auth.py                # Authentication setup
│   ├── config.py              # Config loading / saving / constants
│   ├── guardrails.py          # Crisis detection & topic validation
│   ├── prompts.py             # System & tool prompt templates
│   ├── rate_limiter.py        # Per-user sliding-window rate limiter
│   ├── tools.py               # LangChain tool definitions (5 tools)
│   ├── user_data.py           # User data persistence (JSON per user)
│   ├── utils.py               # PDF extraction & source parsing utilities
│   └── vector_store.py        # Chroma vector store & hybrid retriever
│
├── ui/                        # Frontend modules (one per tab)
│   ├── chat.py                # Chat tab — agent-powered Q&A
│   ├── blood_pressure.py      # Blood Pressure Tracker tab
│   ├── journal.py             # Health Journal tab
│   ├── medical_visit.py       # Medical Visit tab
│   ├── vaccination.py         # Vaccination Records tab
│   ├── prescription.py        # Prescription Manager tab
│   ├── lab_test.py            # Lab Test Records tab
│   ├── documents.py           # Manage Knowledge Base tab
│   └── sidebar.py             # Sidebar — user info, tokens, config
│
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures (MockLLM, ErrorLLM)
│   ├── test_agent.py          # Agent orchestration tests
│   ├── test_config.py         # Configuration tests
│   ├── test_guardrails.py     # Crisis detection & validation tests
│   ├── test_process_query.py  # Query processing pipeline tests
│   ├── test_rate_limiter.py   # Rate limiter tests
│   └── test_utils.py          # Utility function tests
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
│   ├── lung_cancer.pdf
│   └── chroma_db/             # Persisted Chroma database
│
├── user_data/                 # Per-user JSON data files
├── journal_attachments/       # Journal file uploads (per user)
├── lab_reports_samples/       # Sample lab reports for testing
└── assets/                    # Static assets (CSS, images)
    ├── banner.svg
    └── style.css
```

> **Note:** `checkpoints.db` (SQLite memory store) is created at runtime and excluded from version control.

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

# Optional: enable LangSmith tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 3. Run the Application

```bash
streamlit run app.py
```

On first launch, the vector store will be built from the PDFs in `data/` (this may take a few minutes). Subsequent launches load the persisted database instantly.

### 4. Log In

Use one of the demo accounts or click **Create Account** in the app to register a new user:

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
    "temperature": 0,
    "max_context_messages": 20
  },
  "rate_limit": {
    "max_requests": 20,
    "window_seconds": 3600
  },
  "limits": {
    "summary_max_chars": 10000,
    "lab_report_max_chars": 4000
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
| `llm.model` | OpenAI model (e.g. `gpt-4o-mini`, `gpt-4o`) |
| `llm.temperature` | 0 = deterministic, 1 = creative |
| `llm.max_context_messages` | Max recent messages sent to the LLM per turn |
| `rate_limit.max_requests` | Max requests per user per window |
| `rate_limit.window_seconds` | Sliding window duration in seconds |
| `limits.summary_max_chars` | Max characters of document text sent for summarization |
| `limits.lab_report_max_chars` | Max characters of lab report text sent for analysis |

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  Guardrails │──→ Crisis? → Emergency resources
│  (Safety)   │──→ Off-topic? → Redirect (with conversation context)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Rate Limiter│──→ Over limit? → Wait message
└─────┬───────┘
      │
      ▼
┌─────────────────── LangGraph StateGraph ───────────────────┐
│                                                             │
│  ┌────────────┐     ┌──────────────────┐                    │
│  │ agent_node │────→│     tool_node    │                    │
│  │   (LLM)   │     │                  │                    │
│  │            │     │ • search_kb      │                    │
│  │ Decides    │     │ • search_web     │                    │
│  │ which tool │     │ • summarize_doc  │                    │
│  │ to call    │     │ • analyze_lab    │                    │
│  │            │     │ • get_bp_data    │                    │
│  └──────┬─────┘     └────────┬─────────┘                    │
│         │                    │                              │
│         │              ┌─────▼──────────┐                   │
│   No tool calls        │grade_retrieval │                   │
│         │              │                │                   │
│         ▼              │ KB results     │                   │
│        END             │ relevant? ─No──→ Guidance message  │
│                        │   │            │  (try web search) │
│                        │  Yes           │                   │
│                        │   │            │                   │
│                        └───┼────────────┘                   │
│                            │                                │
│                            └──→ back to agent_node          │
│                                                             │
│  ┌──────────────┐                                           │
│  │ SqliteSaver  │  Persistent memory (per-user threads)     │
│  │ Checkpointer │  Saves state after every node execution   │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
   Response with sources & token usage
```

> When LangSmith tracing is enabled, every node in this graph (agent, tools, grade_retrieval) is automatically captured with inputs, outputs, and latency. See the [Observability with LangSmith](#observability-with-langsmith) section to enable it.

## Safety

Health Assistant includes multiple safety layers:

- **Crisis Detection** &mdash; Regex patterns identify medical emergencies (chest pain, stroke symptoms) and mental health crises (suicidal ideation, self-harm). Patterns match both first-person ("I want to end my life") and third-person phrasing ("my friend wants to take their own life"), immediately providing emergency resources (911, 988 Suicide & Crisis Lifeline). Mental health patterns are checked before medical emergency patterns.
- **Health Topic Validation** &mdash; An LLM classifier ensures queries are health-related before processing; uses recent conversation context to correctly handle follow-up queries and document operations; fails open (allows the query through) on any classifier error so legitimate health queries are never silently blocked
- **Content Boundaries** &mdash; The system prompt prohibits diagnosis, medication prescriptions, and treatment plans; all responses include a disclaimer to consult a healthcare professional
- **Rate Limiting** &mdash; Prevents abuse with per-user request throttling; crisis detection runs before the rate limiter so emergency responses are never blocked

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

## Testing

Run the test suite with:

```bash
uv run pytest
```

85 tests across 6 modules, running in under 4 seconds with no API calls:

| Module | Tests | What's covered |
|--------|-------|----------------|
| `test_guardrails.py` | 24 | Crisis detection (mental health, medical emergency, benign inputs, third-person phrasing, priority ordering), `is_health_related` with mock/error LLMs |
| `test_agent.py` | 6 | End-to-end graph execution with fake LLMs — direct answer path and tool-calling path |
| `test_process_query.py` | 9 | Full pipeline orchestration — crisis short-circuit, rate limit blocking, out-of-scope rejection, success path, graph error handling |
| `test_rate_limiter.py` | 15 | In-memory and SQLite backends — boundary conditions, window expiry, user isolation, idempotent init |
| `test_config.py` | 12 | Config load/save round-trip, missing file errors, API key validation including LangSmith misconfiguration detection |
| `test_utils.py` | 12 | `extract_tools_and_sources`, `build_thread_id`, `calculate_session_cost` |

**Testing approach:** LLM calls are replaced with `MockLLM` and `ErrorLLM` fixtures that return predetermined responses, making tests deterministic and free. The end-to-end graph tests use real LangGraph execution with fake LLMs — the closest to a real invocation without hitting any API.

## Deployment

### Railway (recommended)

Railway provides persistent disk storage, which is required for Chroma, SQLite checkpoints, and user data to survive restarts.

1. Create a project at [railway.app](https://railway.app) and connect your GitHub repo
2. Add environment variables in the Variables tab:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_API_KEY=your_serpapi_api_key
   PORT=8501
   ```
3. Attach a persistent volume with mount path `/data`
4. Deploy — the `Procfile` handles symlinking the persistent volume to the app's data paths automatically

On first deploy the vectorstore is built from the PDFs in `data/` and written to the persistent volume. All subsequent deploys load it instantly.

### Streamlit Community Cloud (free, limited persistence)

Suitable for demos and portfolio projects where data persistence across restarts is not required.

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add secrets in the dashboard (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   SERPAPI_API_KEY = "your_serpapi_api_key"
   ```
4. Deploy

**Note:** The `data/chroma_db/` directory must be committed to the repo so the vectorstore loads on startup. User data (blood pressure readings, journal entries, etc.) and conversation history will reset on container restart.

## Observability with LangSmith

[LangSmith](https://smith.langchain.com) is an observability platform built by the LangChain team. When enabled, it automatically captures a full trace of every agent run — every LLM call, tool invocation, retrieval result, and token count — with latency breakdowns and input/output inspection. This requires zero code changes.

### What you can see in LangSmith

- The full agent decision chain for each user query
- Which tools were called and in what order
- Exact inputs and outputs of every LLM call (including the grading node)
- Token usage and latency per node
- Retrieval quality — what documents were returned and whether the grader accepted them

### How to enable it

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key from your account settings
3. Add to your `.env` file (local) or Railway Variables tab (deployed):

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

That's all. On the next app start, every graph invocation appears in your LangSmith dashboard automatically. The app's `validate_api_keys()` will warn at startup if tracing is enabled but the API key is missing.

### Free tier

LangSmith's free tier includes 5,000 traces per month — more than enough for personal use.

## Built With

- [Streamlit](https://streamlit.io/) &mdash; Web interface with Nordic minimalist design
- [LangGraph](https://langchain-ai.github.io/langgraph/) &mdash; Agent orchestration, state management & checkpointing
- [LangChain](https://www.langchain.com/) &mdash; Tools, retrievers & LLM integration
- [ChromaDB](https://www.trychroma.com/) &mdash; Vector storage & semantic search
- [OpenAI](https://openai.com/) &mdash; LLM (GPT-4o-mini) & embeddings
- [SerpAPI](https://serpapi.com/) &mdash; Web search fallback
- [Plotly](https://plotly.com/python/) &mdash; Blood pressure trend charts
- [Ruff](https://docs.astral.sh/ruff/) &mdash; Linting & formatting
