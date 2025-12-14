from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('D:\AI code\Langchain\paper\Attention Is All You Need.pdf')
docs = loader.load()


character_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0,separator='')
result = character_splitter.split_documents(docs)
print(result[2])