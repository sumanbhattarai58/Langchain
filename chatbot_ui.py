import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Title
st.title("🤖 AI Chatbot")

# Initialize the model (cached to avoid reloading)
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.6,
    )
    model = ChatHuggingFace(llm=llm)
    return model

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
for message in st.session_state.messages[1:]:  # Skip system message
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Convert session state messages to LangChain format
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Get response from model
                response = model.invoke(langchain_messages)
                response_content = response.content
                
                # Display response
                st.markdown(response_content)
                
                # Add assistant response to chat history
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