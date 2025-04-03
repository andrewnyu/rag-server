import os
import requests

#langchain
from langchain_community.document_loaders import CSVLoader, PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models import BaseLLM

# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.llms import HuggingFaceHub

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom LLM that communicates with a RunPod-hosted endpoint
class RunPodLLM(BaseLLM):
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT_URL")

    def _call(self, prompt: str, stop=None):
        try:
            response = requests.post(
                url=self.endpoint_url,
                json={"input": prompt},
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("output", "[No output]")
        except Exception as e:
            return f"[Error communicating with RunPod endpoint: {e}]"

class VectorStore:
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url or os.getenv("LLM_ENDPOINT_URL")
        self.upload_dir = Path("uploads")
        self.docs = []
        self.vectorstore = None
        self.qa_chain = None
        self.rag_chain = None
    
    def load_existing_files(self):
        print(f"[INFO] Scanning directory: {self.upload_dir}")
        for file_path in self.upload_dir.glob("*"):
            if file_path.suffix.lower() == ".csv":
                loader = CSVLoader(file_path=str(file_path))
            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                print(f"[SKIPPED] Unsupported file: {file_path.name}")
                continue
            docs = loader.load()
            print(f"[LOADED] {file_path.name} with {len(docs)} documents")
            self.docs.extend(docs)
            
    def split_existing_documents(self):
        print(f"[INFO] Splitting {len(self.docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.docs = splitter.split_documents(self.docs)
        print(f"[INFO] Total chunks created: {len(self.docs)}")

    def build_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        self.vectorstore = Chroma(self.docs, embeddings)
    
    def load_uploaded_file(self, uploaded_file):
        if uploaded_file.suffix.lower() == ".csv":
            loader = CSVLoader(uploaded_file)
        elif uploaded_file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(uploaded_file))
        else:
            print(f"[SKIPPED] Unsupported file: {uploaded_file.name}")
        uploaded_file_doc = loader.load()
        print(f"[LOADED] {uploaded_file.name} with {len(uploaded_file_doc)} documents")
        
        return uploaded_file_doc
    
    def split_document(self, uploaded_file_doc):
        
        chunks = []
        print(f"[INFO] Splitting {len(uploaded_file_doc)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(uploaded_file_doc)
        print(f"[INFO] Total chunks created: {len(chunks)}")
        
        return chunks
        
    def add_documents(self, uploaded_files):
        for file in uploaded_files:
            doc = self.load_uploaded_file(file)
            chunks = self.split_document(doc)
            self.vectorstore.add_documents(chunks)
        
    def build_qa(self):
        print("[INFO] Initializing RetrievalQA chain...")
        llm = RunPodLLM(endpoint_url=self.endpoint_url)
        retriever = self.vectorstore.as_retriever()
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            
        ])
        self.qa_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.qa_chain)
        
    def ask(self, query):
        result = self.rag_chain.invoke({"input": + query})
        print(f"\n[ANSWER] {result['result']}")
        print("[SOURCES]:")
        for doc in result['source_documents']:
            print(" -", doc.metadata.get("source", "Unknown"))
        
        return result
    
    def retrieve(self): 
        pass

def initiate_rag():
    rag = VectorStore(upload_dir="uploads")
    rag.load_existing_files()
    rag.split_existing_documents()
    rag.build_vectorstore()
    rag.build_qa()

if __name__ == "__main__":
    initiate_rag()
            