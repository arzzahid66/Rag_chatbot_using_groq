from fastapi import FastAPI, HTTPException , APIRouter
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import SeleniumURLLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
router = APIRouter()
# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Check for API keys
if not groq_api_key or not google_api_key:
    raise RuntimeError("API keys are not set. Please check your .env file.")

qdrant_key =os.getenv("qdrant_key")
URL = os.getenv("URL")

def send_to_qdrant(final_documents, embeddings):
    qdrant = Qdrant.from_documents(
            final_documents,
            embeddings,
            url=URL,
            prefer_grpc=True,
            api_key=qdrant_key,
            collection_name="mul_uni_scapper"
        )
    return qdrant

# Initialize the chatbot
llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    urls = [
        "https://www.mul.edu.pk/english/about/message-from-the-vice-chancellor/",
        "https://www.mul.edu.pk/english/about/the-minhaj-university-charter-function/",
        "https://www.mul.edu.pk/english/about/mul-an-international-university/",
        "https://www.mul.edu.pk/english/admission/under-graduate-program/",
        "https://www.mul.edu.pk/english/program/associate-degree-in-computer-science/",
        "https://www.mul.edu.pk/downloads/MUL_Prospectus.pdf",
        "https://www.mul.edu.pk/english/program/bs-computer-science/",
        "https://www.mul.edu.pk/english/associate-degrees/",
        "https://www.mul.edu.pk/english/contact-us/"]
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(data[:20])
    send_to_qdrant(final_documents, embeddings)

# FastAPI app
app = FastAPI()

@router.get("/process")
async def process():
    try:
        vector_embedding()
        return {"status": "success", "message": "Data scraped, chunks created, and stored in vector database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
