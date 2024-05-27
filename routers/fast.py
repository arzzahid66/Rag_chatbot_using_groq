import os
import time
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Check for API keys
if not groq_api_key or not google_api_key:
    raise ValueError("API keys are not set. Please check your .env file.")

# Initialize FastAPI app and APIRouter
app = FastAPI()
router = APIRouter()

# Model for request body
class QueryRequest(BaseModel):
    query: str

def vector_embedding():
    qdrant_key =os.getenv("qdrant_key")
    URL = os.getenv("URL")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    client = QdrantClient(
        url=URL,
        api_key=qdrant_key,
    )
    vecto = Qdrant(
        client=client, collection_name="mul_uni_scapper",
        embeddings=embeddings,
    )
    return vecto

vectors = vector_embedding()

def chat_groq(input_query, vectors):
    llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': input_query})
    return response['answer']

@router.post("/chat_qa/")
async def chat_endpoint(query_request: QueryRequest):
    input_query = query_request.query
    if not input_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    answer = chat_groq(input_query, vectors)
    return {"answer": answer}

# Include the router in the app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
