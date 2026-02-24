# rebuild_vectorstore.py

"""
Utility script to rebuild the Chroma vector store from scratch.
Use this when you've:
- Added new PDFs to config.json
- Changed chunking parameters
- Want to refresh the embeddings
"""

from dotenv import load_dotenv
import json

from vector_store import get_or_create_vectorstore


# Load environment variables
load_dotenv()


def rebuild_vectorstore():
    """Force rebuild of the vector store."""
    
    # Load configuration
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    print("\n" + "="*60)
    print("REBUILDING VECTOR STORE")
    print("="*60)
    print(f"\nChroma Directory: {config['chroma_directory']}")
    print(f"Chunk Size: {config['chunking']['chunk_size']}")
    print(f"Chunk Overlap: {config['chunking']['chunk_overlap']}")
    print(f"\nPDFs to process ({len(config['pdf_files'])}):")
    
    for i, pdf in enumerate(config['pdf_files'], 1):
        print(f"  {i}. {pdf}")
    
    print("\n" + "="*60)
    print("Starting rebuild...")
    print("="*60 + "\n")
    
    # Force recreation of vector store
    vectorstore = get_or_create_vectorstore(
        pdf_paths=config["pdf_files"],
        persist_directory=config["chroma_directory"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        force_recreate=True  # This forces rebuild
    )
    
    print("\n" + "="*60)
    print("REBUILD COMPLETE!")
    print("="*60)
    print("\nYour vector store has been rebuilt and saved.")
    print("You can now run main.py to use the updated knowledge base.\n")


if __name__ == "__main__":
    rebuild_vectorstore()
