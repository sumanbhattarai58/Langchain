from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader 

loader = DirectoryLoader(
    path='paper',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.load()

print ((docs[5]))

#lazy load
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)