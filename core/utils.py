"""
Utility functions for the Health Assistant.

Extracts business logic from the UI layer so that ``app.py`` remains
focused on display and user interaction.
"""

import logging
from io import BytesIO

from pypdf import PdfReader
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from core.config import SOURCE_KNOWLEDGE_BASE, SOURCE_URL_PREFIX, TOOL_SEARCH_KB

logger = logging.getLogger(__name__)


def extract_pdf_text(uploaded_file) -> str:
    """Extract text content from an uploaded PDF file.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` object.

    Returns:
        The concatenated text from all pages, or an empty string
        if extraction fails.
    """
    try:
        reader = PdfReader(BytesIO(uploaded_file.getbuffer()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        logger.error("PDF text extraction failed: %s", e)
        return ""


def extract_tools_and_sources(
    messages: list[BaseMessage],
) -> tuple[list[str], list[str]]:
    """Identify tool names and source URLs from a list of messages.

    Scans for ``AIMessage`` objects with tool calls and
    ``ToolMessage`` objects containing ``URL:`` lines.  If no web
    sources are found but ``search_knowledge_base`` was used, the
    source list defaults to ``["Knowledge Base"]``.

    Args:
        messages: The messages to scan (typically the new messages
            from a single agent turn).

    Returns:
        A tuple of ``(tools_used, sources)``.
    """
    tools_used: list[str] = []
    sources: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])
        if isinstance(msg, ToolMessage) and SOURCE_URL_PREFIX in msg.content:
            for line in msg.content.split("\n"):
                if line.strip().startswith(SOURCE_URL_PREFIX):
                    url = line.strip().replace(SOURCE_URL_PREFIX, "").strip()
                    if url:
                        sources.append(url)

    if not sources and TOOL_SEARCH_KB in tools_used:
        sources = [SOURCE_KNOWLEDGE_BASE]

    return tools_used, sources
