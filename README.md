# RAG Server

A Retrieval-Augmented Generation (RAG) system that enhances LLM responses with relevant information from uploaded documents.

## Features

- Upload and process PDF and text documents
- Semantic search using FAISS and sentence transformers
- Clean UI with timing metrics for performance analysis
- Document management with chunk information
- Context toggle for transparency

## Quick Start with Docker

1. Clone the repository:
   ```
   git clone git@github.com:andrewnyu/rag-server.git
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

## Deployment with Deploy Keys

For secure deployments on remote servers, you can use deploy keys to authenticate with the Git repository.

### 1. Generate a deploy key
Run this command to generate a new SSH key:

```
ssh-keygen -t ed25519 -f ~/.ssh/rag_deploy_key -N "" -C "rag-server-deploy"
```

### 2. Create or update your .env file
Create a .env file with your repository URL and deploy key path:

```
GIT_REPO_URL=git@github.com:andrewnyu/rag-server.git
GIT_BRANCH=main
GIT_DEPLOY_KEY_PATH=~/.ssh/rag_deploy_key

# Server settings
SERVER_PORT=8000
SERVER_HOST=0.0.0.0

# LLM endpoint (if using a custom endpoint)
LLM_ENDPOINT_URL=https://your-llm-endpoint-url/query?text=
```

### 3. Add the public key to your Git repository
Copy the public key:

```
cat ~/.ssh/rag_deploy_key.pub
```

Then add it to your repository:
- GitHub: Repository Settings > Deploy keys > Add deploy key
- GitLab: Repository Settings > Repository > Deploy Keys

Make sure to check "Allow write access" if you need to push changes.

### 4. Using the deploy key

#### To clone the repository:
```
GIT_SSH_COMMAND="ssh -i ~/.ssh/rag_deploy_key -o StrictHostKeyChecking=no" git clone git@github.com:andrewnyu/rag-server.git
```

#### To pull from the repository:
```
GIT_SSH_COMMAND="ssh -i ~/.ssh/rag_deploy_key -o StrictHostKeyChecking=no" git pull
```

### 5. Automated updates with the update script
Make the update script executable:

```
chmod +x update-rag-server.sh
```

Run the script to update your deployment:

```
./update-rag-server.sh
```

This script will:
- Clone the repository if it doesn't exist yet
- Pull the latest changes if it already exists
- Rebuild and restart the Docker containers

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

### Default Documents

The system can automatically load documents from a specified folder on startup:

1. Create a folder for your default documents (default is `./default_docs`)
2. Place your text, PDF, Markdown, or CSV files in this folder
3. Set the `DEFAULT_DOCS_FOLDER` in your `.env` file to point to this folder
4. Restart the server to load the documents

This is useful for including reference materials that should always be available to the RAG system.

## Architecture

- **FastAPI**: Web framework for the backend
- **FAISS**: Vector database for efficient similarity search
- **Sentence Transformers**: For generating document embeddings
- **PyPDF2**: For extracting text from PDF files 