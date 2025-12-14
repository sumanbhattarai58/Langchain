from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Suman_resume.pdf")

docs = loader.load()
# print(docs[0].page_content)

print(len(docs))

print(docs[0].metadata)