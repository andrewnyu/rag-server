import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

# Initialize storage for documents and their embeddings
class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective model
        
        # Add some default documents if no documents are uploaded
        self._add_default_documents()
        
    def _add_default_documents(self):
        """Add default documents to the vector store."""
        default_docs = [
        ]
        
        for doc in default_docs:
            self.add_document(doc)
        
        print(f"Added {len(default_docs)} default documents to the vector store.")

    def add_document(self, doc_path_or_text: str):
        """Preprocess and add document to the vector store.
        
        Args:
            doc_path_or_text: Either a file path to a document or the document text itself
        """
        # Check if the input is a file path
        if os.path.isfile(doc_path_or_text):
            # Extract text based on file type
            if doc_path_or_text.lower().endswith('.pdf'):
                doc_text = self._extract_text_from_pdf(doc_path_or_text)
            else:
                # For text files or other formats
                with open(doc_path_or_text, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read()
        else:
            # Assume it's already text
            doc_text = doc_path_or_text
            
        # Skip empty documents
        if not doc_text or doc_text.strip() == "":
            print(f"Warning: Empty document skipped.")
            return
        
        # Split long documents into chunks (simple approach - split by paragraphs)
        chunks = self._chunk_document(doc_text)
        
        # Add each chunk as a separate document
        for chunk in chunks:
            if chunk.strip():
                self.documents.append(chunk)
        
        # Rebuild the index with the new documents
        self._build_index()
        
    def _chunk_document(self, text, max_length=512):
        """Split document into chunks."""
        # Simple chunking by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = ""
        for para in paragraphs:
            if para.strip():
                if len(current_chunk) + len(para) < max_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # If no paragraphs were found, fall back to sentence chunking
        if not chunks:
            sentences = text.split('.')
            current_chunk = ""
            for sentence in sentences:
                if sentence.strip():
                    if len(current_chunk) + len(sentence) < max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # If still no chunks, just use the text as is
        if not chunks and text.strip():
            chunks = [text.strip()]
            
        return chunks
        
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            text = f"[Error processing document: {pdf_path}]"
        return text
    
    def _build_index(self):
        """Build FAISS index from documents."""
        if not self.documents:
            return
            
        # Generate embeddings for all documents
        embeddings = self.model.encode(self.documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(embeddings.astype(np.float32))
        self.embeddings = embeddings

    def retrieve(self, query: str, top_n=3):
        """Retrieve top_n relevant documents based on semantic similarity."""
        if not self.index or not self.documents:
            return ["No documents have been uploaded yet. Please upload some documents first."]
            
        if not query.strip():
            return ["Your query is empty. Please try a more specific question."]
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(top_n, len(self.documents)))
        
        # Get the documents
        results = []
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > 0.2:  # Threshold for relevance
                results.append(self.documents[idx])
        
        if not results:
            return ["No relevant information found for your query. Please try a different question."]
            
        return results
