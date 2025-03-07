import faiss
import numpy as np
import os
from embedder import Embedder
from pathlib import Path

DB_PATH = "db/faiss.index"
os.makedirs("db", exist_ok=True)

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index = self._load_faiss_index()

    def _load_faiss_index(self):
        if os.path.exists(DB_PATH):
            return faiss.read_index(DB_PATH)
        return faiss.IndexFlatL2(self.embedder.dim())

    def add_document(self, file_path):
        """ Convert document to embeddings and store in FAISS """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        embedding = self.embedder.encode(text)
        self.index.add(np.array([embedding], dtype=np.float32))
        faiss.write_index(self.index, DB_PATH)

    def retrieve(self, query, top_k=3):
        """ Retrieve top K relevant documents """
        query_embedding = self.embedder.encode(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        
        if indices[0][0] == -1:  # No match found
            return []

        return [f"Document {i}" for i in indices[0]]  # Replace with actual file refs
