# health_assistant.py

from dotenv import load_dotenv
import os

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# SerpAPI import (official package)
from serpapi import GoogleSearch

# -------------------------
# Load API keys
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# -------------------------
# Load PDF and create RAG retriever
# -------------------------
loader = PyPDFLoader("data/COVID-19.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=3)

# -------------------------
# Define prompt for LLM
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
    model="gpt-4o-mini",
    temperature=0
)

# -------------------------
# RAG chain
# -------------------------
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# -------------------------
# Function to fallback to SerpAPI
# -------------------------
def serpapi_search(query, max_results=3):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    answers = []
    if "organic_results" in result:
        for item in result["organic_results"][:max_results]:
            title = item.get("title")
            snippet = item.get("snippet")
            answers.append(f"{title}\n{snippet}")
    return "\n\n".join(answers) if answers else None

# -------------------------
# Ask a question
# -------------------------
question = input("Ask a question about COVID-19: ")

# First try PDF RAG
response = chain.invoke(question)

answer_text = response.content.strip()

# If RAG says "I don't know", fallback to SerpAPI
if answer_text.lower() in ["i don't know.", "unknown", "no answer"]:
    print("\nNo answer in PDF, searching the web...\n")
    answer_text = serpapi_search(question)

print("\nAnswer:\n")
print(answer_text)
