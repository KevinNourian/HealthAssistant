# vector_store.py

"""
Vector store management using Chroma.
Handles PDF loading, chunking, embedding, and persistence.
"""

import os
import shutil
from typing import Dict, List, Set
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Path Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_source_path(path: str) -> str:
    """Normalize a file path for consistent source-metadata comparison.

    Converts backslashes to forward slashes so that paths stored in the
    Chroma metadata (which may originate on Windows) can be reliably
    compared against paths coming from config.json.
    """
    return os.path.normpath(path).replace("\\", "/")


# Keep a private alias for internal use
_normalize_source_path = normalize_source_path


def _get_ids_by_source(vectorstore: Chroma) -> Dict[str, List[str]]:
    """Return a mapping of *normalized* source path -> list of chunk IDs."""
    result = vectorstore.get(include=["metadatas"])
    source_to_ids: Dict[str, List[str]] = {}
    if result and result.get("ids") and result.get("metadatas"):
        for doc_id, metadata in zip(result["ids"], result["metadatas"]):
            if metadata and "source" in metadata:
                norm = _normalize_source_path(metadata["source"])
                source_to_ids.setdefault(norm, []).append(doc_id)
    return source_to_ids


def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        List of loaded documents
    """
    all_docs = []
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF not found: {pdf_path}")
            continue
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"✓ Loaded {len(docs)} pages from {pdf_path}")
        except Exception as e:
            print(f"✗ Error loading {pdf_path}: {e}")
    
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split documents into smaller chunks.
 
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
    return chunks


def create_chroma_vectorstore(
    chunks: List[Document],
    persist_directory: str,
    embeddings: OpenAIEmbeddings = None
) -> Chroma:
    """
    Create a new Chroma vector store from document chunks.
    
    Args:
        chunks: List of document chunks to embed
        persist_directory: Directory to save the vector store
        embeddings: Embedding model (default: OpenAIEmbeddings())
        
    Returns:
        Chroma vector store instance
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    # Create directory if it doesn't exist
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✓ Created Chroma vector store with {len(chunks)} chunks")
    print(f"✓ Saved to {persist_directory}")
    
    return vectorstore


def load_chroma_vectorstore(
    persist_directory: str,
    embeddings: OpenAIEmbeddings = None
) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    
    Args:
        persist_directory: Directory containing the saved vector store
        embeddings: Embedding model (default: OpenAIEmbeddings())
        
    Returns:
        Chroma vector store instance
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print(f"✓ Loaded Chroma vector store from {persist_directory}")
    
    return vectorstore


def vectorstore_exists(persist_directory: str) -> bool:
    """
    Check if a vector store already exists.
    
    Args:
        persist_directory: Directory to check
        
    Returns:
        True if vector store exists, False otherwise
    """
    # Chroma creates a chroma.sqlite3 file in the persist directory
    chroma_db = os.path.join(persist_directory, "chroma.sqlite3")
    return os.path.exists(chroma_db)


def get_or_create_vectorstore(
    pdf_paths: List[str],
    persist_directory: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force_recreate: bool = False,
    embeddings: OpenAIEmbeddings = None
) -> Chroma:
    """
    Load existing vector store or create a new one from PDFs.
    
    - Checks if vector store exists
    - Loads from disk if available (and not force_recreate)
    - Otherwise, loads PDFs, chunks them, and creates new vector store
    
    Args:
        pdf_paths: List of PDF file paths to process
        persist_directory: Directory to save/load the vector store
        chunk_size: Size of text chunks (default: 500)
        chunk_overlap: Overlap between chunks (default: 50)
        force_recreate: If True, recreate even if exists (default: False)
        embeddings: Embedding model (default: OpenAIEmbeddings())
        
    Returns:
        Chroma vector store instance ready for querying
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    # Try to load existing vector store
    if not force_recreate and vectorstore_exists(persist_directory):
        print("Loading existing vector store...")
        return load_chroma_vectorstore(persist_directory, embeddings)

    # Force recreate: delete old data first to avoid duplicates
    if force_recreate and vectorstore_exists(persist_directory):
        print("Clearing existing vector store for full rebuild...")
        shutil.rmtree(persist_directory, ignore_errors=True)

    # Create new vector store
    print("Creating new vector store...")
    
    # Load PDFs
    docs = load_pdfs(pdf_paths)
    
    if not docs:
        raise ValueError(
            "No documents loaded. Check your PDF paths in config.json"
        )
    
    # Chunk documents
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    
    # Create and save vector store
    vectorstore = create_chroma_vectorstore(chunks, persist_directory, embeddings)
    
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Add / Remove / Sync
# ─────────────────────────────────────────────────────────────────────────────

def add_pdf_to_vectorstore(
    vectorstore: Chroma,
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embeddings: OpenAIEmbeddings = None
) -> None:
    """Embed a single PDF and add its chunks to an existing vectorstore."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    docs = load_pdfs([pdf_path])
    if docs:
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} chunks from {pdf_path}")


def remove_pdf_from_vectorstore(
    vectorstore: Chroma,
    pdf_path: str,
) -> None:
    """Remove all embeddings that originated from *pdf_path*."""
    norm_path = _normalize_source_path(pdf_path)
    source_to_ids = _get_ids_by_source(vectorstore)
    ids_to_delete = source_to_ids.get(norm_path, [])
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        print(f"✓ Removed {len(ids_to_delete)} chunks for {pdf_path}")
    else:
        print(f"⚠ No chunks found for {pdf_path}")


def sync_vectorstore(
    vectorstore: Chroma,
    pdf_paths: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embeddings: OpenAIEmbeddings = None,
) -> None:
    """Bring the vectorstore in sync with *pdf_paths*.

    - PDFs present in *pdf_paths* but missing from the store are embedded.
    - Embeddings whose source PDF is no longer in *pdf_paths* are deleted.
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    desired: Set[str] = {_normalize_source_path(p) for p in pdf_paths}
    source_to_ids = _get_ids_by_source(vectorstore)
    indexed: Set[str] = set(source_to_ids.keys())

    to_add = desired - indexed
    to_remove = indexed - desired

    # Remove stale embeddings
    for source in to_remove:
        ids = source_to_ids[source]
        vectorstore.delete(ids=ids)
        print(f"✓ Sync: removed {len(ids)} chunks for {source}")

    # Embed new documents
    if to_add:
        paths_to_load = [p for p in pdf_paths
                         if _normalize_source_path(p) in to_add]
        docs = load_pdfs(paths_to_load)
        if docs:
            chunks = chunk_documents(docs, chunk_size, chunk_overlap)
            vectorstore.add_documents(chunks)
            print(f"✓ Sync: added {len(chunks)} chunks for {len(paths_to_load)} new PDF(s)")

    if not to_add and not to_remove:
        print("✓ Vectorstore is already in sync.")


def get_retriever(vectorstore: Chroma, k: int = 3):
    """
    Get a retriever from the vector store.

    Args:
        vectorstore: Chroma vector store instance
        k: Number of documents to retrieve (default: 3)

    Returns:
        Retriever configured for similarity search
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


class HybridRetriever(BaseRetriever):
    """Retriever that combines BM25 (keyword) and vector (semantic) search.

    Results from both sub-retrievers are merged using Reciprocal Rank Fusion
    (RRF), which produces a single ranked list that benefits from exact
    keyword matches *and* semantic similarity.
    """

    bm25_retriever: BM25Retriever
    vector_retriever: BaseRetriever
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Retrieve from both sources
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        # --- Reciprocal Rank Fusion (RRF) ---
        # Score each document as  sum( weight / (rank + 60) )  across lists.
        # The constant 60 is the standard RRF smoothing factor.
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 0.5 / (rank + 60)
            doc_lookup[key] = doc

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 0.5 / (rank + 60)
            doc_lookup[key] = doc

        # Sort by descending RRF score and return top-k
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [doc_lookup[k] for k in sorted_keys[: self.k]]


def get_hybrid_retriever(vectorstore: Chroma, k: int = 3) -> HybridRetriever:
    """Get a hybrid retriever combining BM25 (keyword) and vector (semantic) search.

    Fetches all stored chunks from Chroma to build an in-memory BM25 index,
    then combines it with the Chroma vector retriever using Reciprocal Rank
    Fusion (RRF) with equal weighting.

    Args:
        vectorstore: Chroma vector store instance.
        k: Number of documents each sub-retriever should return (default: 3).

    Returns:
        A ``HybridRetriever`` that merges BM25 and vector search results.
    """
    # Fetch all chunks from Chroma to build the BM25 index
    result = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]

    # BM25: keyword/lexical retriever (in-memory)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Vector: semantic retriever backed by Chroma
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        k=k,
    )
