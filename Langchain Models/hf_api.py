from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    max_new_tokens=200,
    
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("Explain the theory of relativity in simple terms.")
print(result.content)