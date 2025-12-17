from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#step 1: create a list of source documents
documents = [
    Document(page_content = "Protein is the bodybuilding of life"),
    Document(page_content = "gym helps you build muscels in bodybuilding"),
    Document(page_content  = "Protein makes muscles from your body"),
    Document(page_content = "Gym is a place to workout"),
    Document(page_content = "gym helps you build muscles in bodybuilding"),

]

#Step 2:Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vector store in memory
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
    
)

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "what helps you build muscles in bodybuilding?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


