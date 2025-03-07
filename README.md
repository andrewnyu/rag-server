# RAG System

A Retrieval-Augmented Generation (RAG) system that enhances LLM responses with relevant information from uploaded documents.

## Features

- Upload and process PDF and text documents
- Semantic search using FAISS and sentence transformers
- Clean UI with timing metrics for performance analysis
- Document management with chunk information
- Context toggle for transparency

## Docker Setup

### Prerequisites

- Docker and Docker Compose installed on your system

### Running with Docker

1. Clone this repository:
   ```
   git clone <repository-url>
   cd rag-server
   ```

2. Build and start the container:
   ```
   docker-compose up -d
   ```

3. Access the application:
   Open your browser and navigate to `http://localhost:8000`

4. To stop the container:
   ```
   docker-compose down
   ```

### Persistent Data

Uploaded documents are stored in the `uploads` directory, which is mounted as a volume in the Docker container. This ensures that your documents persist even if the container is restarted.

## Project Structure

```
rag-server/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── main.py
├── llm.py
├── vectorstore.py
├── requirements.txt
├── templates/
│   └── index.html
└── uploads/
```

## Development Setup

If you prefer to run the application without Docker:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m nltk.downloader punkt
   ```

2. Run the server with auto-reload:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage

1. Upload documents using the "Documents" tab
2. Switch to the "Chat" tab to ask questions
3. View timing metrics for each query
4. Toggle the context visibility to see what information was used

## Architecture

- **FastAPI**: Web framework for the backend
- **FAISS**: Vector database for efficient similarity search
- **Sentence Transformers**: For generating document embeddings
- **PyPDF2**: For extracting text from PDF files 