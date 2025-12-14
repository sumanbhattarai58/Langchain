from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)
parser = StrOutputParser()

# Load webpage
url = "https://www.daraz.com.np/products/electric-barbeque-grill-and-barbeque-grill-toaster-multifunction-i103030212-s1023799422.html?pvid=0ff2c157-bc33-466e-8557-3b7aa50a48ab&scm=1007.51705.446532.0&spm=a2a0e.tm80335409.just4u.d_103030212"
loader = WebBaseLoader(url)
docs = loader.load()

# Check loaded text
print("Loaded length:", len(docs[0].page_content))

# Format prompt and run model
input_text = prompt.format(
    question='What is the product that we are talking about?',
    text=docs[0].page_content
)
output = model.invoke(input_text)
parsed_output = parser.parse(output)

print(parsed_output)
