from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Load PDF
loader = PyPDFLoader("data/Future of Jobs.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=3)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Ask
question = input("Ask a question about COVID-19: ")
response = chain.invoke(question)

print("\nAnswer:\n", response.content)