from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import time
from llm import LLM
from vectorstore import VectorStore
from dotenv import load_dotenv, set_key
import re
import gc  # Add garbage collection

# Load environment variables
load_dotenv()

app = FastAPI()
# Initialize LLM with endpoint from environment
llm = LLM(endpoint_url=os.getenv("LLM_ENDPOINT_URL"))
vector_store = VectorStore()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Update LLM endpoint
@app.get("/admin/update-llm-endpoint/")
async def update_llm_endpoint(endpoint_url: str = Query(..., description="New LLM endpoint URL")):
    global llm
    
    try:
        # Validate URL format
        if not re.match(r'^https?://.+', endpoint_url):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid URL format. URL must start with http:// or https://"}
            )
        
        # Create a new LLM instance with the updated endpoint
        new_llm = LLM(endpoint_url=endpoint_url)
        
        # Test the endpoint with a simple query
        try:
            test_response = new_llm.generate("test", ["This is a test document"])
            
            # Check if the response indicates an error
            if test_response and test_response.startswith("Error:"):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Endpoint test failed: {test_response}"}
                )
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to test endpoint: {str(e)}"}
            )
        
        # If we get here, the endpoint is working, so update the global LLM instance
        llm = new_llm
        
        # Update the .env file if it exists
        env_file = ".env"
        if os.path.exists(env_file):
            set_key(env_file, "LLM_ENDPOINT_URL", endpoint_url)
        else:
            # Create a new .env file with the endpoint
            with open(env_file, "w") as f:
                f.write(f"LLM_ENDPOINT_URL={endpoint_url}\n")
        
        return JSONResponse(content={
            "message": "LLM endpoint updated successfully",
            "endpoint": endpoint_url
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to update LLM endpoint: {str(e)}"}
        )

# Get current LLM endpoint
@app.get("/admin/llm-endpoint/")
async def get_llm_endpoint():
    return JSONResponse(content={
        "endpoint": llm.endpoint_url
    })

# Add documents to vector store
@app.post("/index-documents/")
async def index_documents(files: list[UploadFile]):
    start_time = time.time()
    
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    uploaded_files = []
    processing_times = []
    
    print(f"Starting to process {len(files)} documents...")
    
    for file in files:
        try:
            doc_start_time = time.time()
            print(f"Processing document: {file.filename}")
            
            file_path = f"uploads/{file.filename}"
            
            # Use chunked copying to avoid loading entire file into memory
            with open(file_path, "wb") as buffer:
                # Copy in chunks of 64KB instead of all at once
                chunk_size = 64 * 1024  # 64KB chunks
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
                
            print(f"File saved to {file_path}, adding to vector store...")
            
            # Add document to vector store with source
            vector_store.add_document(file_path, source=file.filename)
            
            # Force garbage collection after processing each file
            gc.collect()
            
            doc_process_time = time.time() - doc_start_time
            processing_times.append((file.filename, doc_process_time))
            
            uploaded_files.append(file.filename)
            print(f"Successfully added document: {file.filename} in {doc_process_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing document {file.filename}: {str(e)}")
        finally:
            file.file.close()  # Explicitly close the file handle
            # Force garbage collection
            gc.collect()

    process_time = time.time() - start_time
    
    # Print summary
    print(f"Indexing complete. Total time: {process_time:.2f} seconds")
    for filename, doc_time in processing_times:
        print(f"  - {filename}: {doc_time:.2f} seconds")
    
    # Final garbage collection
    gc.collect()
    
    return JSONResponse(content={
        "message": f"Successfully indexed {len(uploaded_files)} files in vector store!",
        "files": uploaded_files,
        "upload_time": f"{process_time:.2f} seconds",
        "file_details": [{"filename": f, "process_time": f"{t:.2f} seconds"} for f, t in processing_times]
    })

# Get list of documents
@app.get("/documents/")
async def get_documents():
    # Get document list from vector store
    document_list = vector_store.get_document_list()
    
    # Add file system information for uploaded documents
    for doc in document_list:
        if doc["filename"] != "Default Example":
            file_path = os.path.join("uploads", doc["filename"])
            if os.path.exists(file_path):
                # Update with actual file size
                doc["size"] = f"{os.path.getsize(file_path) / 1024:.1f} KB"
                # Add last modified time
                doc["last_modified"] = time.ctime(os.path.getmtime(file_path))
            else:
                print(f"Warning: Document file not found: {file_path}")
    
    return JSONResponse(content={"documents": document_list})

# Get list of files in the upload queue (files in uploads folder not yet indexed)
@app.get("/upload-queue/")
async def get_upload_queue():
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Get all files in the uploads directory
    uploaded_files = []
    indexed_files = [doc["filename"] for doc in vector_store.get_document_list() 
                    if doc["filename"] != "Default Example"]
    
    for filename in os.listdir("uploads"):
        file_path = os.path.join("uploads", filename)
        if os.path.isfile(file_path):
            # Check if file is already indexed in vector store
            if filename not in indexed_files:
                # Get file info
                file_size = os.path.getsize(file_path)
                last_modified = time.ctime(os.path.getmtime(file_path))
                
                # Determine file type from extension
                file_type = "UNKNOWN"
                if "." in filename:
                    extension = filename.split(".")[-1].upper()
                    if extension:
                        file_type = extension
                
                uploaded_files.append({
                    "filename": filename,
                    "size": f"{file_size / 1024:.1f} KB",
                    "last_modified": last_modified,
                    "type": file_type
                })
    
    return JSONResponse(content={"files": uploaded_files})


#helper function to extract the answer from the LLM response
def extract_answer(llm_response):
    marker = "Answer:"
    if marker in llm_response:
        return llm_response.split(marker, 1)[-1].strip()
    else:
        return llm_response.strip()

# Ask a question
@app.get("/ask/")
async def ask_question(query: str = Query(..., title="User query")):
    # Start timing
    start_time = time.time()
    
    # Retrieve relevant documents
    retrieval_start = time.time()

    expanded_query = llm.expand_query(query)

    docs = vector_store.retrieve(expanded_query)
    retrieval_time = time.time() - retrieval_start
    
    # Generate answer using the LLM
    llm_start = time.time()
    answer = llm.generate(expanded_query, docs)

    try:
        answer = str(answer).split("Question:")[1] #manually remove the context from the answer
    except IndexError:
        pass
    
    llm_time = time.time() - llm_start
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Extract just the content for display in the UI
    context_for_display = []
    for doc in docs:
        # Extract source and content from the formatted string
        if isinstance(doc, str):
            parts = doc.split("\n\n", 1)
            if len(parts) > 1:
                source = parts[0].replace("[Document ", "").split("]")[0]
                content = parts[1]
                context_for_display.append({
                    "source": source,
                    "content": content[:300] + "..." if len(content) > 300 else content
                })

    
    clean_answer = extract_answer(answer)
    
    # Return comprehensive response with timing data
    return JSONResponse(content={
        "answer": answer,
        "context": context_for_display,
        "timing": {
            "retrieval_time": f"{retrieval_time:.2f} seconds",
            "llm_time": f"{llm_time:.2f} seconds",
            "total_time": f"{total_time:.2f} seconds"
        }
    })

# Upload files to the queue (without indexing)
@app.post("/upload-only/")
async def upload_files_only(files: list[UploadFile]):
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    uploaded_files = []
    
    for file in files:
        try:
            file_path = f"uploads/{file.filename}"
            # Check if file already exists
            if os.path.exists(file_path):
                continue
            
            # Use chunked copying to avoid loading entire file into memory
            with open(file_path, "wb") as buffer:
                # Copy in chunks of 64KB instead of all at once
                chunk_size = 64 * 1024  # 64KB chunks
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
            
            uploaded_files.append(file.filename)
            print(f"File uploaded to queue: {file.filename}")
        finally:
            file.file.close()  # Explicitly close the file handle
            gc.collect()  # Force garbage collection
    
    return JSONResponse(content={
        "message": f"Successfully uploaded {len(uploaded_files)} files to queue",
        "files": uploaded_files
    })

# Remove a file from the upload queue
@app.delete("/remove-from-queue/")
async def remove_from_queue(filename: str):
    file_path = f"uploads/{filename}"
    
    # Check if file exists
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found in upload queue"}
        )
    
    # Check if it's already indexed
    indexed_files = [doc["filename"] for doc in vector_store.get_document_list() 
                   if doc["filename"] != "Default Example"]
    
    if filename in indexed_files:
        return JSONResponse(
            status_code=400,
            content={"error": "Cannot remove file as it's already indexed in the vector store"}
        )
    
    # Remove the file
    try:
        os.remove(file_path)
        return JSONResponse(content={"message": f"File {filename} removed from queue"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to remove file: {str(e)}"}
        )

# Index files that are already in the upload folder
@app.post("/index-files/")
async def index_files(request: Request):
    data = await request.json()
    filenames = data.get("filenames", [])
    
    if not filenames:
        return JSONResponse(
            status_code=400,
            content={"error": "No filenames provided for indexing"}
        )
    
    start_time = time.time()
    
    indexed_files = []
    processing_times = []
    
    print(f"Starting to process {len(filenames)} files from upload queue...")
    
    for filename in filenames:
        try:
            doc_start_time = time.time()
            print(f"Processing file: {filename}")
            
            file_path = f"uploads/{filename}"
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print(f"Adding file to vector store: {filename}")
            
            # Add document to vector store with memory optimization
            # Process in smaller batches to prevent memory buildup
            vector_store.add_document(file_path, source=filename)
            
            # Force garbage collection after processing each file
            gc.collect()
            
            doc_process_time = time.time() - doc_start_time
            processing_times.append((filename, doc_process_time))
            
            indexed_files.append(filename)
            print(f"Successfully indexed file: {filename} in {doc_process_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error indexing {filename}: {str(e)}")
            # Force garbage collection after an error
            gc.collect()
    
    process_time = time.time() - start_time
    
    # Print summary
    print(f"Indexing complete. Total time: {process_time:.2f} seconds")
    for filename, doc_time in processing_times:
        print(f"  - {filename}: {doc_time:.2f} seconds")
    
    # Final garbage collection
    gc.collect()
    
    return JSONResponse(content={
        "message": f"Successfully indexed {len(indexed_files)} files in vector store!",
        "files": indexed_files,
        "process_time": f"{process_time:.2f} seconds",
        "file_details": [{"filename": f, "process_time": f"{t:.2f} seconds"} for f, t in processing_times]
    })
