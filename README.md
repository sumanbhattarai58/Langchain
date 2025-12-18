# AI Chatbot – YouTube Transcript Q&A

An AI-powered chatbot that answers questions from YouTube videos. Users provide a YouTube URL id, and the chatbot retrieves the transcript of the video using **YouTubeTranscriptAPI**.
It uses **LangChain** and **RAG (Retrieval-Augmented Generation)** to process the transcript and generate **context-aware answers** via an LLM. The application is deployed on **Streamlit** for 
interactive demonstrations.

**Key Highlights:**
- Uses **YouTubeTranscriptAPI** to fetch accurate video transcripts.(Video with only english subtitle/caption)
- Implements **RAG** to find relevant information in long transcripts.
- Uses **LangChain** for conversation logic, chaining queries and responses with context.
- Deployed with **Streamlit**, providing an easy-to-use interface for live demos.
- Supports questions in **real-time** without needing to pre-process videos manually.


**[Watch Demo Video]**
https://drive.google.com/file/d/1hoYmz1QCOqvOtuZ43I_MtgO6ROgCSADR/view?usp=drive_link


## Technologies
- Python
- LangChain
- RAG (Retrieval-Augmented Generation)
- Streamlit
- HuggingFace
