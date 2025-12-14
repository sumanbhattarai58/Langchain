from langchain_huggingface import HuggingFaceEmbeddings
'''
Imports the LangChain wrapper for HuggingFace embeddings.

HuggingFaceEmbeddings lets you load a sentence-transformers model and generate embeddings (vector representations) for text.
'''


from dotenv import load_dotenv
'''
Imports the dotenv loader, which lets you read environment variables from a .env file.
Often used to load API keys or secrets.
'''


from sklearn.metrics.pairwise import cosine_similarity
'''
Imports cosine similarity function from scikit-learn.
Cosine similarity measures how similar two vectors are in vector space.
Values range from -1 (opposite) to 1 (identical). For embeddings, higher means more semantically similar.
'''


import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Suman Bhattarai is a data scientist working in the field of AI and machine learning.",
    "Sujan Bhattarai is a software engineer with expertise in web development and cloud computing.",
    "Hari Bastola is electrical engineer working in embedded systems and IoT.",
    "Shankar Bastola is a DevOps engineer specializing in CI/CD pipelines and infrastructure as code.",
]

query = 'tell me about Suman Bhattarai'

doc_embeddings = embedding.embed_documents(documents)
'''
Converts all documents into vector embeddings using the model.
doc_embeddings is now a list of 384-dimensional vectors, one for each document.
'''

query_embedding = embedding.embed_query(query)
'''
Converts the query into a vector embedding using the same model.
Ensures the query and documents are in the same vector space for comparison.
'''

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
'''
query_embedding is 1D, shape (384,).
cosine_similarity expects a 2D array, shape (n_samples, n_features).
'''

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



