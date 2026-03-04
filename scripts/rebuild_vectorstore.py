"""
Utility script to rebuild the Chroma vector store from scratch.
Use this when you've:
- Added new PDFs to config.json
- Changed chunking parameters
- Want to refresh the embeddings
"""

import json
import os
import sys

from dotenv import load_dotenv

# Allow running from any directory by adding project root
_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)
sys.path.insert(0, _ROOT)

from core.vector_store import get_or_create_vectorstore  # noqa: E402


# Load environment variables
load_dotenv(os.path.join(_ROOT, ".env"))


def rebuild_vectorstore():
    """Force rebuild of the vector store."""

    # Load configuration
    config_path = os.path.join(_ROOT, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print("\n" + "=" * 60)
    print("REBUILDING VECTOR STORE")
    print("=" * 60)
    print(
        f"\nChroma Directory: "
        f"{config['chroma_directory']}"
    )
    print(
        f"Chunk Size: "
        f"{config['chunking']['chunk_size']}"
    )
    print(
        f"Chunk Overlap: "
        f"{config['chunking']['chunk_overlap']}"
    )
    print(
        f"\nPDFs to process "
        f"({len(config['pdf_files'])}):"
    )

    for i, pdf in enumerate(config['pdf_files'], 1):
        print(f"  {i}. {pdf}")

    print("\n" + "=" * 60)
    print("Starting rebuild...")
    print("=" * 60 + "\n")

    # Force recreation of vector store
    vectorstore = get_or_create_vectorstore(
        pdf_paths=config["pdf_files"],
        persist_directory=config["chroma_directory"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        force_recreate=True,
    )

    print("\n" + "=" * 60)
    print("REBUILD COMPLETE!")
    print("=" * 60)
    print("\nYour vector store has been rebuilt and saved.")
    print(
        "You can now run app.py to use the "
        "updated knowledge base.\n"
    )


if __name__ == "__main__":
    rebuild_vectorstore()
