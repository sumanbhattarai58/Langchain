from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 1. Load web document
# -----------------------------
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.worldwildlife.org/species/tiger/"
loader = WebBaseLoader(url)
docs = loader.load()

print(f"Loaded documents: {len(docs)}")

# -----------------------------
# 2. Split text into chunks
# -----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# 3. Create embeddings
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 4. Store chunks in Chroma (Vector DB)
# -----------------------------
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------
query = "What does a tiger eat?"

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)



# -----------------------------
# 6. Load LLM (HuggingFace)
# -----------------------------
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# 7. Ask question using retrieved context
# -----------------------------
context = "\n\n".join(doc.page_content[:800] for doc in relevant_docs)


prompt = f"""

Context:
{context}

Question:
What do a tiger eat?
"""

response = model.invoke(prompt)

print("\n--- FINAL ANSWER ---\n")
print(response)
