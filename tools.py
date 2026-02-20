"""
LangChain tool definitions for the Health Assistant agent.

Tools are created via a factory function so that dependencies (vectorstore,
retriever, LLM, config) are injected rather than accessed as globals.
"""

import os

from langchain_core.tools import tool
from serpapi import GoogleSearch

from config import SUMMARY_MAX_CHARS, LAB_REPORT_MAX_CHARS
from prompts import SUMMARY_PROMPT, LAB_ANALYSIS_PROMPT
from vector_store import normalize_source_path


def create_tools(vectorstore, retriever, llm, config: dict, serpapi_key: str) -> list:
    """Create and return all agent tools with injected dependencies.

    Args:
        vectorstore: The Chroma vector store instance.
        retriever: The LangChain retriever for similarity search.
        llm: The ChatOpenAI LLM instance.
        config: Application configuration dict.
        serpapi_key: API key for SerpAPI web search.

    Returns:
        A list of LangChain tool objects.
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the health knowledge base for information from uploaded
        medical documents. Use this tool when the user asks a health question
        that might be answered by their personal document library."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def search_web(query: str) -> str:
        """Search the web for health information. Use this when the knowledge
        base does not contain the answer, or the user explicitly asks for web
        results."""
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
        }
        try:
            result = GoogleSearch(params).get_dict()
            snippets = []
            for item in result.get("organic_results", [])[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if title and snippet:
                    snippets.append(f"**{title}**\n{snippet}\nURL: {link}")
            return "\n\n".join(snippets) if snippets else "No web results found."
        except Exception:
            return "Web search failed. Please try again."

    @tool
    def summarize_document(filename: str) -> str:
        """Summarize a specific PDF document from the knowledge base. The user
        may refer to the document by its filename (e.g. 'nutrition_guide.pdf').
        Use this tool when the user asks for a summary or overview of a
        document."""
        # Find matching PDF path from config
        matched_path = None
        for pdf_path in config["pdf_files"]:
            if filename.lower() in os.path.basename(pdf_path).lower():
                matched_path = pdf_path
                break

        if not matched_path:
            available = ", ".join(
                os.path.basename(p) for p in config["pdf_files"]
            )
            return (
                f"Document '{filename}' not found. "
                f"Available documents: {available}"
            )

        try:
            normalized = normalize_source_path(matched_path)
            docs = vectorstore.similarity_search(
                "summary of document", k=10, filter={"source": normalized}
            )
            if not docs:
                docs = vectorstore.similarity_search(
                    "summary of document", k=10,
                    filter={"source": matched_path}
                )
            if not docs:
                return (
                    f"No content found for {os.path.basename(matched_path)}"
                )

            combined_text = "\n\n".join(doc.page_content for doc in docs)
            prompt = SUMMARY_PROMPT.format(
                content=combined_text[:SUMMARY_MAX_CHARS]
            )
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    @tool
    def analyze_lab_report(report_text: str) -> str:
        """Analyze a medical lab report. The report_text parameter should
        contain the extracted text from a lab report PDF. Use this tool when
        the user uploads a lab report and asks for analysis of their
        results."""
        if not report_text.strip():
            return "The report appears to be empty or could not be read."

        prompt = LAB_ANALYSIS_PROMPT.format(
            report_text=report_text[:LAB_REPORT_MAX_CHARS]
        )
        response = llm.invoke(prompt)
        return response.content

    return [search_knowledge_base, search_web, summarize_document, analyze_lab_report]
