from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace
from langchain_classic.retrievers import MultiQueryRetriever

documents = [
    Document(
        page_content="Protein is essential for muscle growth and bodybuilding. It helps repair and build muscle tissues after workouts.",
        metadata={"source": "fitness/protein.txt"}
    ),
    Document(
        page_content="Regular gym workouts help build muscles, increase strength, and improve overall bodybuilding performance.",
        metadata={"source": "fitness/gym.txt"}
    ),
    Document(
        page_content="Muscle building requires a combination of resistance training, adequate protein intake, and proper recovery.",
        metadata={"source": "fitness/muscle_building.txt"}
    ),
    Document(
        page_content="Bodybuilding focuses on increasing muscle size through structured training programs and proper nutrition.",
        metadata={"source": "fitness/bodybuilding.txt"}
    ),
    Document(
        page_content="Recovery, sleep, and nutrition play a major role in muscle development and bodybuilding success.",
        metadata={"source": "fitness/recovery.txt"}
    ),
]

# Initialize Huggingface embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Faiss vector store
vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type = 'mmr', search_kwargs={"k": 2}),
    llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-1.5B-Instruct"))
)
# Query
query = "What helps in building muscles for bodybuilding?"

multiquery_results= multiquery_retriever.invoke(query)


for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)