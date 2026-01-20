# vector_store.py

"""
Vector store management using Chroma.
Handles PDF loading, chunking, embedding, and persistence.
"""

import os
from typing import List
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
    
    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked documents
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
    
    This is the main function you'll use. It handles the entire workflow:
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
