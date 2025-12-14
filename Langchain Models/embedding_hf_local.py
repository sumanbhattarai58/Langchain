from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "The theory of relativity is a set of scientific theories that describe the relationship between gravity and space and time."

vector = embedding.embed_query(text)
print(str(vector))