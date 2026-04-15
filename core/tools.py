"""
LangChain tool definitions for the Health Assistant agent.

Tools are created via a factory function so that dependencies (vectorstore,
retriever, LLM, config) are injected rather than accessed as globals.  This
makes the tools easier to test and avoids hidden coupling.

Each tool returns a plain string: either the useful result or a
human-readable error message.  Exceptions are caught inside each tool
so that the agent loop never crashes mid-conversation.
"""

import logging
import os
from typing import Any

from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from tenacity import retry, stop_after_attempt, wait_exponential

from core.prompts import SUMMARY_PROMPT, LAB_ANALYSIS_PROMPT
from core.vector_store import normalize_source_path

logger = logging.getLogger(__name__)


def create_tools(
    vectorstore: Any,
    retriever: Any,
    llm: ChatOpenAI,
    config: dict[str, Any],
    serpapi_key: str,
    bp_readings_ref: list[dict[str, Any]] | None = None,
) -> list[BaseTool]:
    """Create and return all agent tools with injected dependencies.

    Args:
        vectorstore: The Chroma vector store instance.
        retriever: The LangChain retriever for similarity search.
        llm: The ChatOpenAI LLM instance.
        config: Application configuration dictionary.
        serpapi_key: API key for SerpAPI web search.
        bp_readings_ref: Reference to the user's blood pressure
            readings list.  Because Python lists are mutable, this
            always reflects the latest data.

    Returns:
        A list of LangChain ``BaseTool`` objects ready to be bound to
        the LLM.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the health knowledge base for information
        from uploaded medical documents. Use this tool when
        the user asks a health question that might be
        answered by their personal document library."""
        
        logger.info("search_knowledge_base called with query: %s", query)
        try:
            # Query rewriting: rephrase the conversational query into a
            # concise, keyword-rich retrieval query before hitting the
            # vector store.
            rewrite_prompt = (
                "Rewrite the following question into a concise, keyword-rich "
                "search query for a medical document retrieval system. "
                "Remove conversational language. Return only the rewritten "
                "query, nothing else.\n\nQuestion: " + query
            )
            rewritten_query = llm.invoke(rewrite_prompt).content.strip()
            logger.info("Rewritten query: %s", rewritten_query)

            docs = retriever.invoke(rewritten_query)
            if not docs:
                logger.info("No documents matched the query")
                return "No relevant information found in the knowledge base."
            logger.info("Retrieved %d document chunks", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error("Knowledge base search failed: %s", e)
            return f"Knowledge base search encountered an error: {e}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _serpapi_search(params: dict[str, str]) -> dict:
        """Execute a SerpAPI search with retry on transient failures.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s…
        capped at 10s).  ``reraise=True`` ensures the final exception
        propagates to the caller's try/except if all attempts fail.

        Args:
            params: The SerpAPI query parameters.

        Returns:
            The raw search results dictionary.
        """
        return GoogleSearch(params).get_dict()

    @tool
    def search_web(query: str) -> str:
        """Search the web for health information. Use this when the knowledge
        base does not contain the answer, or the user explicitly asks for web
        results."""
        logger.info("search_web called with query: %s", query)
        params: dict[str, str] = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
        }
        try:
            result = _serpapi_search(params)
            snippets: list[str] = []
            for item in result.get("organic_results", [])[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if title and snippet:
                    snippets.append(
                        f"**{title}**\n{snippet}\nURL: {link}"
                    )
            if not snippets:
                logger.info("Web search returned no usable results")
                return "No web results found."
            logger.info("Web search returned %d results", len(snippets))
            return "\n\n".join(snippets)
        except Exception as e:
            logger.error("Web search failed after retries: %s", e)
            return f"Web search failed: {e}"

    @tool
    def summarize_document(filename: str) -> str:
        """Summarize a specific PDF document from the
        knowledge base. The user may refer to the document
        by its filename (e.g. 'nutrition_guide.pdf'). Use
        this tool when the user asks for a summary or
        overview of a document."""
        logger.info("summarize_document called for: %s", filename)

        # Find matching PDF path from config
        matched_path: str | None = None
        for pdf_path in config["pdf_files"]:
            if filename.lower() in os.path.basename(pdf_path).lower():
                matched_path = pdf_path
                break

        if not matched_path:
            available = ", ".join(
                os.path.basename(p) for p in config["pdf_files"]
            )
            logger.warning("Document '%s' not found in config", filename)
            return (
                f"Document '{filename}' not found. "
                f"Available documents: {available}"
            )

        try:
            normalized = normalize_source_path(matched_path)
            docs = vectorstore.similarity_search(
                "summary of document", k=25, filter={"source": normalized}
            )
            if not docs:
                docs = vectorstore.similarity_search(
                    "summary of document",
                    k=25,
                    filter={"source": matched_path},
                )
            if not docs:
                logger.warning("No chunks found for %s", matched_path)
                return (
                    f"No content found for "
                    f"{os.path.basename(matched_path)}"
                )

            # Count total chunks for this document to inform the user.
            all_chunks = vectorstore.get(
                where={"source": normalized}
            )
            total_chunks = len(all_chunks.get("ids", []))
            if total_chunks == 0:
                all_chunks = vectorstore.get(
                    where={"source": matched_path}
                )
                total_chunks = len(all_chunks.get("ids", []))

            combined_text = "\n\n".join(doc.page_content for doc in docs)
            prompt = SUMMARY_PROMPT.format(
                content=combined_text[:config["limits"]["summary_max_chars"]]
            )
            logger.info(
                "Summarizing %d of %d chunks (%d chars)",
                len(docs),
                total_chunks,
                len(combined_text),
            )
            response = llm.invoke(prompt)
            summary = response.content

            # Add transparency note if the summary is based on a subset.
            if total_chunks > len(docs):
                summary += (
                    f"\n\n---\n*Note: This summary is based on "
                    f"{len(docs)} of {total_chunks} sections from "
                    f"the document and may not cover all content.*"
                )

            return summary
        except Exception as e:
            logger.error("Summarization failed for '%s': %s", filename, e)
            return f"Error generating summary: {e}"

    @tool
    def analyze_lab_report(report_text: str) -> str:
        """Analyze a medical lab report. The report_text parameter should
        contain the extracted text from a lab report PDF. Use this tool when
        the user uploads a lab report and asks for analysis of their
        results."""
        logger.info(
            "analyze_lab_report called (%d chars of text)",
            len(report_text),
        )
        if not report_text.strip():
            logger.warning("Empty report text received")
            return "The report appears to be empty or could not be read."

        try:
            prompt = LAB_ANALYSIS_PROMPT.format(
                report_text=report_text[
                    :config["limits"]["lab_report_max_chars"]
                ]
            )
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error("Lab report analysis failed: %s", e)
            return f"Error analyzing lab report: {e}"

    @tool
    def get_blood_pressure_data(query: str) -> str:
        """Retrieve the user's blood pressure readings. Use this tool when
        the user asks about their blood pressure history, trends, or
        averages. Readings include systolic, diastolic, and pulse data.
        The query parameter can describe the time period of interest
        (e.g. 'last month', 'all readings')."""
        logger.info("get_blood_pressure_data called with query: %s", query)
        if bp_readings_ref is None:
            return "Blood pressure tracking is not available."
        try:
            if not bp_readings_ref:
                return "No blood pressure readings recorded yet."
            lines: list[str] = []
            for r in bp_readings_ref:
                pulse_str = (
                    f", pulse {r['pulse']} BPM"
                    if r.get("pulse") else ""
                )
                lines.append(
                    f"{r['date']} {r.get('time', '')} — "
                    f"{r['systolic']}/{r['diastolic']} mmHg"
                    f"{pulse_str}"
                )
            logger.info(
                "Returning %d blood pressure readings", len(bp_readings_ref)
            )
            return (
                f"Blood pressure readings ({len(bp_readings_ref)} total):\n"
                + "\n".join(lines)
                + "\n\nWhen presenting these readings, include a brief "
                "interpretation for each using standard AHA categories: "
                "Normal (<120/<80), Elevated (120-129/<80), "
                "High Blood Pressure Stage 1 (130-139 or 80-89), "
                "High Blood Pressure Stage 2 (≥140 or ≥90), "
                "Hypertensive Crisis (>180 and/or >120). "
                "Also note any trends (improving, worsening, stable). "
                "If pulse is present, note if it is within a normal "
                "resting range (60-100 BPM). "
                "End with a reminder that this is general guidance "
                "and not a substitute for professional medical advice."
            )
        except Exception as e:
            logger.error("Blood pressure data retrieval failed: %s", e)
            return f"Error retrieving blood pressure data: {e}"

    return [
        search_knowledge_base,
        search_web,
        summarize_document,
        analyze_lab_report,
        get_blood_pressure_data,
    ]
