import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import PyPDF2
import docx
import io

load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Chatbot : Your friendly chatbot", page_icon="🤖", layout="centered")

# Title
st.title("🤖 AI Chatbot")

# Initialize the model (cached to avoid reloading)
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="EssentialAI/rnj-1-instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5,
    )
    model = ChatHuggingFace(llm=llm)
    return model

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    elif file_type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    
    else:
        return None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant"}
    ]

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Make sure you have set HUGGINGFACEHUB_API_TOKEN in your .env file")
    st.stop()

# Display chat history (excluding system message)
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File upload section with icon
col1, col2 = st.columns([5, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📎",
        type=["pdf", "docx", "txt"],
        label_visibility="visible",
        help="Upload a file (PDF, DOCX, or TXT) to get a summary"
    )

# Handle file upload
if uploaded_file is not None:
    with st.spinner("Processing file..."):
        extracted_text = extract_text_from_file(uploaded_file)
        
        if extracted_text:
            # Truncate if too long (adjust as needed)
            if len(extracted_text) > 3000:
                extracted_text = extracted_text[:3000] + "..."
            
            # Auto-generate summary request
            summary_prompt = f"Please summarize the following document:\n\n{extracted_text}"
            
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded file: {uploaded_file.name}"})
            
            with st.chat_message("user"):
                st.markdown(f"📄 Uploaded file: {uploaded_file.name}")
            
            # Generate summary
            with st.chat_message("assistant"):
                with st.spinner("Generating summary..."):
                    try:
                        langchain_messages = []
                        for msg in st.session_state.messages:
                            if msg["role"] == "system":
                                langchain_messages.append(SystemMessage(content=msg["content"]))
                            elif msg["role"] == "user":
                                langchain_messages.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                langchain_messages.append(AIMessage(content=msg["content"]))
                        
                        # Add summary request
                        langchain_messages.append(HumanMessage(content=summary_prompt))
                        
                        response = model.invoke(langchain_messages)
                        response_content = response.content
                        
                        st.markdown(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        else:
            st.error("Could not extract text from the file. Please try a different file format.")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                response = model.invoke(langchain_messages)
                response_content = response.content
                
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Sidebar with options
with st.sidebar:
    st.header("Options")
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant"}
        ]
        st.rerun()
    
    st.divider()
    
    st.subheader("Chat Statistics")
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    ai_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    st.metric("Your Messages", user_messages)
    st.metric("AI Messages", ai_messages)
    
    st.divider()
    
    st.caption("Powered by HuggingFace 🤗")