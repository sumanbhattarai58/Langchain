from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "automobiles.txt",
    encoding="utf-8-sig"
)

docs = loader.load()
print(type(docs))

print(len(docs))

print(docs[0])



