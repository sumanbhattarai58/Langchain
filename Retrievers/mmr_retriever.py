from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content = "Protein is the bodybuilding of life"),
    Document(page_content = "gym helps you build muscels in bodybuilding"),
    Document(page_content  = "Protein makes muscles from your body"),
    Document(page_content = "Gym is a place to workout"),
    Document(page_content = "gym helps you build muscles in bodybuilding"),
]

from langchain_community.vectorstores import FAISS

# Initialize OpenAI embeddings
embedding_model = HuggingFaceEmbeddings()

# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)

query = "What is the bodybuilding of life"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)