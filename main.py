from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from llm import LLM
from vectorstore import VectorStore

app = FastAPI()
llm = LLM(endpoint_url="http://your-gpu-endpoint-url/query?text=")  
vector_store = VectorStore()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")

# Upload documents
@app.post("/upload/")
async def upload_file(files: list[UploadFile]):
    # Ensure uploads directory exists
    os.makedirs("rag-server/uploads", exist_ok=True)
    
    for file in files:
        file_path = f"rag-server/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        vector_store.add_document(file_path)

    return JSONResponse(content={"message": "Files uploaded successfully!"})

# Ask a question
@app.get("/ask/")
async def ask_question(query: str = Query(..., title="User query")):
    docs = vector_store.retrieve(query)
    answer = llm.generate(query, docs)
    return JSONResponse(content={"answer": answer})
