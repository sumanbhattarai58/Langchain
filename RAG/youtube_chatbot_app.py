import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.title("🎥 Interact with the YouTube video you want to")

# Initialize session state for storing processed video data
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Video ID Input Section
st.header("Step 1: Load Video")
video_id = st.text_input(
    "Enter the YouTube video ID (not the full URL, but just the ID)",
    placeholder="e.g., UclrVWafRAI",
    help="The video ID is the part after 'v=' in the YouTube URL"
)

if st.button("Load Video Transcript"):
    if video_id:
        with st.spinner("Fetching and processing transcript..."):
            try:
                # Fetch transcript
                transcript_list = YouTubeTranscriptApi().fetch(video_id)
                transcript = " ".join([chunk.text for chunk in transcript_list])
                
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.create_documents([transcript])
                
                # Create embeddings and vector store
                embedding = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
                vector_store = FAISS.from_documents(chunks, embedding)
                
                # Store in session state
                st.session_state.vector_store = vector_store
                st.session_state.current_video_id = video_id
                
                # ✅ FIX: Clear chat history when a new video is successfully loaded
                st.session_state.chat_history = []
                
                st.success(f"✅ Transcript loaded successfully! ({len(chunks)} chunks created)")
                st.info(f"📺 Video: https://www.youtube.com/watch?v={video_id}")
                
            except TranscriptsDisabled:
                st.error("❌ No captions available for this video.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a video ID first.")

# Question Section
st.header("Step 2: Ask Questions")

if st.session_state.vector_store is not None:

    # Display previous Q&A
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**🧑 Question {i+1}:** {chat['question']}")
        st.markdown(f"**🤖 Answer:** {chat['answer']}")
        st.divider()

    # New question input (always fresh)
    # We add the length to the key to ensure unique keys, but since we clear history on load, 
    # it resets to 0 which works perfectly.
    question = st.text_input(
        "Ask a new question:",
        key=f"question_{len(st.session_state.chat_history)}",
        placeholder="e.g., What dangers will human face from AI in the future?"
    )

    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    # ---- YOUR EXISTING LOGIC (UNCHANGED) ----
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type='similarity',
                        search_kwargs={'k': 3}
                    )

                    def format_docs(retrieved_docs):
                        return "\n\n".join(doc.page_content for doc in retrieved_docs)

                    prompt = PromptTemplate(
                        template="""
                        You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.

                        {context}
                        Question: {question}
                        """,
                        input_variables=['context', 'question']
                    )

                    llm = HuggingFaceEndpoint(
                        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
                        task="text-generation",
                        max_new_tokens=400,
                        temperature=0.2,
                        top_p=0.9,
                        repetition_penalty=1.1,
                    )
                    model = ChatHuggingFace(llm=llm)

                    parallel_chain = RunnableParallel({
                        'context': retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                    })

                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt | model | parser

                    result = main_chain.invoke(question)

                    # ✅ Store Q&A instead of overwriting UI
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result
                    })

                    st.rerun()  # 🔥 This creates the new section automatically

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("👆 Please load a video transcript first before asking questions.")


# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Find the YouTube video ID:**
       - From URL: `youtube.com/watch?v=**UclrVWafRAI**`
       - The ID is: `UclrVWafRAI`
    
    2. **Enter the ID** in the input box
    
    3. **Click "Load Video Transcript"**
    
    4. **Ask questions** about the video content
    
    5. **Get AI-powered answers** based on the transcript
    """)
    
    if st.session_state.current_video_id:
        st.success(f"✅ Current Video: {st.session_state.current_video_id}")