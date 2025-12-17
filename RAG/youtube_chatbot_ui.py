from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

video_id = "UclrVWafRAI" # only the ID, not full URL
try:
    
    transcript_list = YouTubeTranscriptApi().fetch(video_id)

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=10)
chunks = splitter.create_documents([transcript])

embedding = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(chunks, embedding)

retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k':1})


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question= "what danger do human have from ai in the future"
retrieved_docs    = retriever.invoke(question)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()

llm= HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    max_new_tokens=800,
    
)

model = ChatHuggingFace(llm=llm)



main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke(question)
print(result)