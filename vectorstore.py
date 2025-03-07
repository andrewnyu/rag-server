import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2

# Initialize storage for documents and their embeddings
class VectorStore:
    def __init__(self):
        self.documents = []
        self.document_sources = []  # Track document sources
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
            self.add_document(doc, source="Default Example")
        
        print(f"Added {len(default_docs)} default documents to the vector store.")

    def add_document(self, doc_path_or_text: str, source=None):
        """Preprocess and add document to the vector store.
        
        Args:
            doc_path_or_text: Either a file path to a document or the document text itself
            source: Source of the document (e.g., filename or description)
        """
        # Check if the input is a file path
        if os.path.isfile(doc_path_or_text):
            # Extract text based on file type
            if doc_path_or_text.lower().endswith('.pdf'):
                doc_text = self._extract_text_from_pdf(doc_path_or_text)
                source = source or os.path.basename(doc_path_or_text)
            else:
                # For text files or other formats
                with open(doc_path_or_text, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read()
                source = source or os.path.basename(doc_path_or_text)
        else:
            # Assume it's already text
            doc_text = doc_path_or_text
            source = source or "Direct Text Input"
            
        # Skip empty documents
        if not doc_text or doc_text.strip() == "":
            print(f"Warning: Empty document skipped.")
            return
        
        # Split long documents into chunks (simple approach - split by paragraphs)
        chunks = self._chunk_document(doc_text)
        
        # Add each chunk as a separate document
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                self.documents.append(chunk)
                # For chunks, add source with chunk number
                if len(chunks) > 1:
                    self.document_sources.append(f"{source} (Chunk {i+1}/{len(chunks)})")
                else:
                    self.document_sources.append(source)
        
        # Rebuild the index with the new documents
        self._build_index()
        
    def _chunk_document(self, text, max_length=512, overlap=100):
        """Split document into chunks with overlap to preserve context.
        
        Args:
            text: The document text to chunk
            max_length: Maximum chunk size in characters
            overlap: Number of characters to overlap between chunks
        """
        # First try to split by paragraphs
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # If paragraphs are too long, we'll need to split them further
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            para_length = len(para)
            
            # If paragraph fits in current chunk, add it
            if current_length + para_length <= max_length:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_length += para_length + 2  # +2 for the newlines
            
            # If paragraph is too big for a single chunk, split it by sentences
            elif para_length > max_length:
                # First add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_length = 0
                
                # Split paragraph into sentences
                sentences = [s.strip() + "." for s in para.split('.') if s.strip()]
                
                # Process sentences with overlap
                i = 0
                while i < len(sentences):
                    current_chunk = ""
                    current_length = 0
                    
                    # Add sentences until we reach max_length
                    while i < len(sentences) and current_length + len(sentences[i]) <= max_length:
                        if current_chunk:
                            current_chunk += " " + sentences[i]
                        else:
                            current_chunk = sentences[i]
                        current_length += len(sentences[i]) + 1  # +1 for the space
                        i += 1
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Move back for overlap (but not below 0)
                    overlap_sentences = max(1, int(overlap / 30))  # Approximate number of sentences for desired overlap
                    i = max(0, i - overlap_sentences)
            
            # If paragraph doesn't fit, start a new chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                current_length = para_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no chunks were created (rare case), just use the original text
        if not chunks and text.strip():
            # Try to split into smaller pieces if text is very large
            if len(text) > max_length:
                # Simple character-based chunking with overlap as fallback
                chunks = []
                for i in range(0, len(text), max_length - overlap):
                    chunks.append(text[i:i + max_length].strip())
            else:
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

    def retrieve(self, query: str, top_n=5, threshold=0.2):
        """Retrieve top_n relevant documents based on semantic similarity.
        
        Args:
            query: The user query
            top_n: Number of documents to retrieve
            threshold: Minimum similarity score threshold
        
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.index or not self.documents:
            return ["No documents have been uploaded yet. Please upload some documents first."]
        
        if not query.strip():
            return ["Your query is empty. Please try a more specific question."]
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents - get more than needed for filtering
        search_k = min(top_n * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        # Get the documents with their metadata
        results = []
        seen_sources = set()  # To track unique sources
        
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > threshold:  # Apply similarity threshold
                doc = self.documents[idx]
                source = self.document_sources[idx]
                base_source = source.split(" (Chunk")[0]
                
                # Add source diversity - prefer different sources
                if len(results) >= top_n and base_source in seen_sources:
                    continue
                    
                seen_sources.add(base_source)
                
                # Add metadata to help with context
                results.append({
                    "content": doc,
                    "source": source,
                    "score": float(scores[0][i]),
                    "relevance": "High" if scores[0][i] > 0.6 else "Medium" if scores[0][i] > 0.4 else "Low"
                })
                
                # Stop once we have enough results
                if len(results) >= top_n:
                    break
        
        if not results:
            return ["No relevant information found for your query. Please try a different question."]
        
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Format results for the LLM
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(
                f"[Document {i+1}] {res['source']} (Relevance: {res['relevance']})\n\n{res['content']}"
            )
        
        return formatted_results
        
    def get_document_list(self):
        """Get a list of unique documents in the vector store."""
        if not self.documents or not self.document_sources:
            return []
            
        # Create a list of unique documents by source
        unique_sources = set()
        document_list = []
        
        for i, source in enumerate(self.document_sources):
            # Extract the base source name (without chunk info)
            base_source = source.split(" (Chunk")[0]
            
            if base_source not in unique_sources:
                unique_sources.add(base_source)
                
                # Get document type
                doc_type = "TEXT"
                if "." in base_source:
                    extension = base_source.split(".")[-1].upper()
                    if extension:
                        doc_type = extension
                
                # Get document size (approximate based on first chunk)
                doc_size = len(self.documents[i]) / 1024  # Size in KB
                
                # Count chunks for this document
                chunk_count = sum(1 for s in self.document_sources if s.startswith(base_source))
                
                document_list.append({
                    "filename": base_source,
                    "type": doc_type,
                    "size": f"{doc_size:.1f} KB",
                    "chunks": chunk_count
                })
                
        return document_list
