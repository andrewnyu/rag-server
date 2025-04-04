import os
import requests
from pathlib import Path
from typing import Optional, List, Union

from dotenv import load_dotenv
from loguru import logger

from langchain_community.document_loaders import CSVLoader, PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load .env vars
load_dotenv()


class RunPodLLM(BaseLLM):
    """Custom LLM that communicates with a RunPod-hosted endpoint."""
    endpoint_url: str = os.getenv("LLM_ENDPOINT_URL")
    if not endpoint_url:
        raise ValueError("Environment variable 'LLM_ENDPOINT_URL' is not set.")

    def _llm_type(self) -> str:
        return "runpod-llama"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        headers = {
        # "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}",
        "Content-Type": "application/json"
        }
        for prompt in prompts:
            try:
                logger.debug(f"Sending prompt to LLM: {prompt[:80]}...")
                payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.5,
                    "stop": stop if stop else []
                    }
                }
                response = requests.post(
                    self.endpoint_url.rstrip("/") + "/query?text=",
                    headers=headers,
                    json=payload,
                    # timeout=60,
                    verify=False    # Disable SSL verification for local testing
                )
                response.raise_for_status()
                try:
                    output = response.json().get("generated_text", "[No output]")
                except requests.exceptions.JSONDecodeError:
                    logger.error("Failed to decode JSON response")
                    output = "[Invalid JSON response]"
            except Exception as e:
                logger.error(f"RunPod error: {e}")
                output = f"[RunPod error: {str(e)}]"
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)


class VectorStore:
    def __init__(self, upload_dir: Union[str, Path] = "uploads", endpoint_url=None):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT_URL") + "/query?text="
        self.upload_dir = Path(upload_dir)
        self.docs = []
        self.vectorstore = None
        self.qa_chain = None
        self.rag_chain = None

    def _load_file(self, file_path: Path):
        ext = file_path.suffix.lower()
        loader = None
        if ext == ".csv":
            loader = CSVLoader(file_path=str(file_path))
        elif ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {file_path.name}")
            return []
        try:
            docs = loader.load()
            logger.info(f"Loaded {file_path.name} with {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            return []

    def load_existing_files(self):
        logger.info(f"Scanning directory: {self.upload_dir}")
        for file_path in self.upload_dir.glob("*"):
            docs = self._load_file(file_path)
            self.docs.extend(docs)

    def split_existing_documents(self):
        logger.info(f"Splitting {len(self.docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.docs = splitter.split_documents(self.docs)
        logger.success(f"Total chunks created: {len(self.docs)}")

    def build_vectorstore(self):
        logger.info("Building vectorstore with embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=embeddings,
            collection_name="my_collection"
        )
        logger.success("Vectorstore built successfully.")

    def add_documents(self, uploaded_files: List[Path]):
        for file in uploaded_files:
            docs = self._load_file(file)
            if not docs:
                continue
            chunks = self._split_documents(docs)
            self.vectorstore.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks from {file.name}")

    def _split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    def build_qa(self):
        logger.info("Setting up Retrieval-Augmented Generation (RAG) chain...")
        llm = llm = RunPodLLM()
        
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant for question-answering tasks. Use the retrieved context below which contains the product list of our company."
             "Provide possible product suggestions based on the query and the context."
             "If you don't know the answer, say so. Be concise.\n\n{context}"),
            ("human", "{input}")
        ])

        self.qa_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.qa_chain)
        logger.success("RAG QA chain initialized.")

    def ask(self, query: str):
        try:   
            logger.info(f"Querying: {query}")
            result = self.rag_chain.invoke({"input": query})
            logger.info(f"Answer: {result.get('result', '[No answer]')}")
            logger.info("Sources:")
            for doc in result.get('source_documents', []):
                print(" -", doc.metadata.get("source", "Unknown"))
            return result
    
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {"result": f"[Error] {str(e)}", "source_documents": []}
        
    def retrieve(self, query: str):
        if not self.vectorstore:
            logger.error("Vectorstore is not initialized. Please build the vectorstore first.")
            return {"error": "Vectorstore is not initialized."}
        return self.vectorstore.similarity_search(query, k=10)


def initiate_rag() -> VectorStore:
    logger.info("Initializing RAG pipeline...")
    rag = VectorStore(upload_dir="uploads")
    rag.load_existing_files()
    rag.split_existing_documents()
    rag.build_vectorstore()
    rag.build_qa()
    logger.success("RAG pipeline ready.")
    return rag


if __name__ == "__main__":
    pass