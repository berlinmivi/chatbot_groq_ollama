from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins to limit requests from certain domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# State variables (you can initialize once globally)
embeddings = OllamaEmbeddings(model="mistral:latest")
loader = WebBaseLoader("https://docs.llamaindex.ai/en/stable/")# You can use any url you want.
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Initialize Prompt Template and Chain
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Define input model for the chat endpoint
class ChatRequest(BaseModel):
    prompt: str


# Define response model for the chat endpoint
class ChatResponse(BaseModel):
    answer: str
    context: List[str]


@app.post("/chat", response_model=ChatResponse)
def get_response(request: ChatRequest):
    """
    Chat endpoint for generating a response based on the prompt and context.
    """
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": request.prompt})
        print("Response time: ", time.process_time() - start)

        return ChatResponse(
            answer=response["answer"],
            context=[doc.page_content for doc in response["context"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[str])
def get_documents():
    """
    Endpoint to get loaded document chunks (for debugging or exploration).
    """
    return [doc.page_content for doc in final_documents]


@app.post("/reload_documents")
def reload_documents():
    """
    Endpoint to reload the documents and embeddings.
    """
    global docs, final_documents, vectors, retriever, retrieval_chain

    try:
        docs = loader.load()
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return {"status": "Documents reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)