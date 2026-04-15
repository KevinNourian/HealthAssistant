"""Manage Documents tab UI — knowledge base management."""

import logging
import os
from typing import Any

import streamlit as st

from core.config import save_config
from core.vector_store import (
    add_pdf_to_vectorstore,
    remove_pdf_from_vectorstore,
)

logger = logging.getLogger(__name__)


def render(
    username: str,
    config: dict[str, Any],
    vectorstore: Any,
) -> None:
    """Render the Manage Documents tab.

    Args:
        username: The authenticated username.
        config: The application configuration dictionary.
        vectorstore: The Chroma vector store instance.
    """
    st.markdown("### Manage Knowledge Base")
    st.caption("Upload PDFs to expand your health knowledge base")

    st.markdown("#### Current Documents")

    if config["pdf_files"]:
        for idx, pdf_path in enumerate(config["pdf_files"]):
            col1, col2 = st.columns([5, 1])

            with col1:
                filename: str = os.path.basename(pdf_path)
                file_size: str = "Unknown size"
                if os.path.exists(pdf_path):
                    size_bytes: int = os.path.getsize(pdf_path)
                    size_kb: float = size_bytes / 1024
                    if size_kb < 1024:
                        file_size = f"{size_kb:.1f} KB"
                    else:
                        file_size = f"{size_kb / 1024:.1f} MB"

                st.markdown(f"📄 **{filename}** · {file_size}")

            with col2:
                if st.button(
                    "Delete", key=f"del_pdf_{idx}",
                    use_container_width=True,
                ):
                    try:
                        logger.info("Deleting document: %s", pdf_path)
                        remove_pdf_from_vectorstore(vectorstore, pdf_path)
                        st.session_state.pdf_files.remove(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

                        st.cache_resource.clear()
                        logger.info("Document deleted successfully")
                    except Exception as e:
                        logger.error("Error deleting document: %s", e)
                        st.error(f"Error deleting file: {e}")

                    st.rerun()
    else:
        st.info("No documents in knowledge base yet")

    st.markdown("---")

    st.markdown("#### Add New Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="kb_pdf_upload",
        help=(
            "Upload health-related PDF documents to add "
            "to your knowledge base"
        ),
    )

    if uploaded_pdf:
        st.success(f"✓ Selected: **{uploaded_pdf.name}**")

        if st.button(
            "Add to Knowledge Base", type="primary",
            use_container_width=True,
        ):
            add_success: bool = False
            with st.spinner("Adding document to knowledge base..."):
                try:
                    pdf_dir: str = "data"
                    os.makedirs(pdf_dir, exist_ok=True)

                    pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)
                    logger.info("Saving uploaded PDF to %s", pdf_path)

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    if pdf_path not in st.session_state.pdf_files:
                        st.session_state.pdf_files.append(pdf_path)
                        config["pdf_files"] = st.session_state.pdf_files
                        save_config(config)

                    add_pdf_to_vectorstore(
                        vectorstore,
                        pdf_path,
                        chunk_size=config["chunking"]["chunk_size"],
                        chunk_overlap=config["chunking"]["chunk_overlap"],
                    )

                    st.cache_resource.clear()
                    add_success = True
                    logger.info("Document added successfully: %s", pdf_path)
                except Exception as e:
                    logger.error("Error adding document: %s", e)
                    st.error(f"❌ Error adding document: {e}")

            if add_success:
                st.rerun()
    else:
        st.info("👆 Choose a PDF file to add to your knowledge base")
