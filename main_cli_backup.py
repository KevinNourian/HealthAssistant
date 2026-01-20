# main.py

"""
Health Assistant - RAG-based Q&A system with web fallback.
Uses Chroma vector store for PDF knowledge base.
"""

from dotenv import load_dotenv
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from serpapi import GoogleSearch

from vector_store import get_or_create_vectorstore, get_retriever


# -------------------------
# Load API keys
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# -------------------------
# Load configuration
# -------------------------
def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


config = load_config()


# -------------------------
# Initialize Vector Store
# -------------------------
print("\n" + "="*60)
print("Initializing Health Assistant...")
print("="*60 + "\n")

# Get or create vector store
vectorstore = get_or_create_vectorstore(
    pdf_paths=config["pdf_files"],
    persist_directory=config["chroma_directory"],
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_overlap"],
    force_recreate=False  # Set to True to rebuild from scratch
)

# Get retriever
retriever = get_retriever(vectorstore, k=config["retriever"]["k"])


# -------------------------
# Define RAG prompt
# -------------------------
prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)


# -------------------------
# Initialize LLM
# -------------------------
llm = ChatOpenAI(
    model=config["llm"]["model"],
    temperature=config["llm"]["temperature"]
)


# -------------------------
# Create RAG chain
# -------------------------
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)


# -------------------------
# Web search fallback
# -------------------------
def serpapi_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using SerpAPI when PDF doesn't have the answer.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    
    try:
        search = GoogleSearch(params)
        result = search.get_dict()
        answers = []
        
        if "organic_results" in result:
            for item in result["organic_results"][:max_results]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if title and snippet:
                    answers.append(f"{title}\n{snippet}")
        
        return "\n\n".join(answers) if answers else "No results found."
    
    except Exception as e:
        return f"Search error: {e}"


# -------------------------
# Main interaction loop
# -------------------------
def main():
    """Run the health assistant."""
    print("\n" + "="*60)
    print("Health Assistant Ready!")
    print("="*60)
    print("Ask questions about COVID-19 (or other topics in your PDFs)")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        # Get user question
        question = input("Your question: ").strip()
        
        # Check for exit
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using Health Assistant. Goodbye!")
            break
        
        # Skip empty questions
        if not question:
            continue
        
        try:
            # Query the RAG system
            print("\n🔍 Searching knowledge base...")
            response = chain.invoke(question)
            answer_text = response.content.strip()
            
            # Check if answer was found
            if answer_text.lower() in ["i don't know.", "i don't know", "unknown"]:
                print("📡 No answer in PDFs, searching the web...\n")
                answer_text = serpapi_search(question)
                source = "Web Search"
            else:
                source = "PDF Knowledge Base"
            
            # Display answer
            print(f"\n{'='*60}")
            print(f"Source: {source}")
            print(f"{'='*60}\n")
            print(answer_text)
            print(f"\n{'-'*60}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
