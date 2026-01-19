# Health Assistant - RAG Application

A simple, modular RAG (Retrieval-Augmented Generation) application for querying PDF documents with automatic web fallback.

## 🎯 Features

- **Function-based architecture** - Simple, easy to understand and modify
- **Chroma vector store** - Persistent, local, and efficient
- **One-time embedding** - PDFs are processed once and saved to disk
- **Multi-PDF support** - Query across multiple documents
- **Web fallback** - Automatically searches the web if answer isn't in PDFs
- **Clean configuration** - JSON-based config for easy customization

## 📁 Project Structure

```
.
├── main.py                  # Main application
├── vector_store.py          # Vector store functions (Chroma)
├── config.json              # Configuration file
├── rebuild_vectorstore.py   # Utility to rebuild embeddings
├── requirements.txt         # Dependencies
├── .env                     # API keys (create this)
└── data/
    ├── COVID-19.pdf         # Your PDF files
    └── chroma_db/           # Chroma database (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### 3. Add Your PDFs

Place your PDF files in the `data/` directory and update `config.json`:

```json
{
  "pdf_files": [
    "data/COVID-19.pdf",
    "data/health-guide.pdf"
  ]
}
```

### 4. Run the Application

```bash
python main.py
```

**First run:** Creates embeddings (may take a few minutes)  
**Subsequent runs:** Loads from disk (instant!)

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
  "pdf_files": [
    "data/COVID-19.pdf"
  ],
  "chroma_directory": "data/chroma_db",
  "chunking": {
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "retriever": {
    "k": 3
  },
  "llm": {
    "model": "gpt-4o-mini",
    "temperature": 0
  }
}
```

### Configuration Options:

- **`pdf_files`**: List of PDF paths to include in knowledge base
- **`chroma_directory`**: Where to save the vector database
- **`chunk_size`**: Maximum characters per chunk (smaller = more precise, larger = more context)
- **`chunk_overlap`**: Characters to overlap between chunks (helps maintain context)
- **`retriever.k`**: Number of relevant chunks to retrieve per query
- **`llm.model`**: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4")
- **`llm.temperature`**: 0 = deterministic, 1 = creative

## 📝 Usage Examples

### Basic Q&A

```
Your question: What are the symptoms of COVID-19?

🔍 Searching knowledge base...

============================================================
Source: PDF Knowledge Base
============================================================

The main symptoms include fever, dry cough, fatigue, and 
shortness of breath...

------------------------------------------------------------
```

### Web Fallback

```
Your question: What is the latest COVID variant in 2024?

🔍 Searching knowledge base...
📡 No answer in PDFs, searching the web...

============================================================
Source: Web Search
============================================================

CDC Reports New COVID Variant JN.1
The latest variant shows increased transmissibility...

------------------------------------------------------------
```

## 🔄 Updating Your Knowledge Base

### Adding New PDFs

1. Place new PDFs in `data/` directory
2. Add paths to `config.json`:
   ```json
   {
     "pdf_files": [
       "data/COVID-19.pdf",
       "data/new-research.pdf"
     ]
   }
   ```
3. Rebuild the vector store:
   ```bash
   python rebuild_vectorstore.py
   ```

### Changing Chunking Parameters

If you modify `chunk_size` or `chunk_overlap` in `config.json`, rebuild:

```bash
python rebuild_vectorstore.py
```

## 🏗️ Architecture Overview

### How It Works

1. **Initialization** → Checks if Chroma database exists
2. **Load or Create** → Loads existing database OR creates new from PDFs
3. **Query** → User asks a question
4. **Retrieve** → Finds relevant chunks from vector store
5. **Generate** → LLM generates answer from retrieved context
6. **Fallback** → If no answer found, searches the web

### Key Functions in `vector_store.py`

```python
# Main function - handles everything
vectorstore = get_or_create_vectorstore(
    pdf_paths=["data/file.pdf"],
    persist_directory="data/chroma_db",
    force_recreate=False
)

# Individual functions if you need more control
docs = load_pdfs(pdf_paths)
chunks = chunk_documents(docs, chunk_size=500)
vectorstore = create_chroma_vectorstore(chunks, persist_dir)
retriever = get_retriever(vectorstore, k=3)
```

## 🛠️ Extending the Application

### Adding More PDF Types

The `PyPDFLoader` works with standard PDFs. For other formats:

```python
# In vector_store.py, modify load_pdfs():
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
```

### Using a Different LLM

Edit `config.json`:

```json
{
  "llm": {
    "model": "gpt-4",
    "temperature": 0.3
  }
}
```

### Switching to a Different Vector Store

While this version uses Chroma, you can easily switch to another vector store by modifying `vector_store.py`. The function-based design makes this straightforward:

1. Replace Chroma imports
2. Update `create_*_vectorstore()` function
3. Update `load_*_vectorstore()` function
4. Keep the same function signatures

## 🎓 Why This Architecture?

### Simple & Maintainable
- **Functions over classes** → Easier to understand and modify
- **Clear flow** → Load → Chunk → Embed → Save → Query
- **Minimal abstraction** → No over-engineering

### Efficient
- **One-time processing** → Embeddings cached on disk
- **Fast startup** → Loads from Chroma in seconds
- **Automatic persistence** → Chroma handles saving

### Flexible
- **Easy to extend** → Add new functions without breaking existing code
- **Modular** → Replace components independently
- **Configurable** → JSON config for non-code changes

## 📚 Common Issues

### "No documents loaded"
- Check PDF paths in `config.json` are correct
- Ensure PDF files exist in the specified locations

### "Chroma database not found"
- First run creates the database
- Or manually run: `python rebuild_vectorstore.py`

### Slow first run
- Normal! Embedding creation takes time
- Subsequent runs load from disk (fast)

### API errors
- Verify `.env` file exists with valid API keys
- Check API key permissions and quotas

## 📄 License

MIT License - feel free to use and modify for your projects!

## 🤝 Contributing

This is a simple educational project. Feel free to:
- Add more features
- Improve error handling
- Support more document types
- Add better logging
- Create a Streamlit UI

---

**Built with:** LangChain, Chroma, OpenAI, SerpAPI
