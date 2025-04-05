import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import glob
from dotenv import load_dotenv
import gc  # Add garbage collection

# Load environment variables
load_dotenv()

# Initialize storage for documents and their embeddings
class VectorStore:
    def __init__(self):
        self.documents = []
        self.document_sources = []  # Track document sources
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective model
        
        # Add some default documents if no documents are uploaded
        self._add_default_documents()
        
    def _add_default_documents(self):
        """Add default documents to the vector store from the folder specified in .env."""
        # Get default docs folder from environment variable or use default
        default_docs_folder = os.getenv('DEFAULT_DOCS_FOLDER', './default_docs')
        
        # Create the folder if it doesn't exist
        os.makedirs(default_docs_folder, exist_ok=True)
        
        # Find all files in the default docs folder
        default_docs = []
        supported_extensions = ['.txt', '.pdf', '.md', '.csv']
        
        for ext in supported_extensions:
            default_docs.extend(glob.glob(os.path.join(default_docs_folder, f'*{ext}')))
        
        # Add each document to the vector store
        for doc_path in default_docs:
            try:
                self.add_document(doc_path, source=f"Default: {os.path.basename(doc_path)}")
                print(f"Added default document: {doc_path}")
            except Exception as e:
                print(f"Error adding default document {doc_path}: {e}")
        
        print(f"Added {len(default_docs)} default documents to the vector store.")

    def add_document(self, doc_path_or_text: str, source=None):
        """Preprocess and add document to the vector store.
        
        Args:
            doc_path_or_text: Either a file path to a document or the document text itself
            source: Source of the document (e.g., filename or description)
        """
        print(f"Processing document: {source or doc_path_or_text}")
        
        # Check if the input is a file path
        if os.path.isfile(doc_path_or_text):
            # Extract text based on file type
            if doc_path_or_text.lower().endswith('.pdf'):
                print("Extracting text from PDF...")
                doc_text = self._extract_text_from_pdf(doc_path_or_text)
                source = source or os.path.basename(doc_path_or_text)
            else:
                # For text files or other formats - process in chunks to save memory
                print("Reading text file...")
                doc_text = ""
                with open(doc_path_or_text, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read file in chunks of 1MB
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        doc_text += chunk
                source = source or os.path.basename(doc_path_or_text)
        else:
            # Assume it's already text
            doc_text = doc_path_or_text
            source = source or "Direct Text Input"
            
        # Skip empty documents
        if not doc_text or doc_text.strip() == "":
            print(f"Warning: Empty document skipped.")
            return
            
        text_length = len(doc_text)
        print(f"Document size: {text_length} characters")
        
        # Split long documents into chunks using the simpler algorithm
        print(f"Chunking document...")
        chunks = self.simple_chunk(doc_text)
        print(f"Created {len(chunks)} chunks")
        
        # Clear doc_text from memory
        doc_text = None
        gc.collect()
        
        # Add each chunk as a separate document
        print(f"Adding chunks to document store...")
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                self.documents.append(chunk)
                # For chunks, add source with chunk number
                if len(chunks) > 1:
                    self.document_sources.append(f"{source} (Chunk {i+1}/{len(chunks)})")
                else:
                    self.document_sources.append(source)
        
        # Clear chunks from memory
        chunks = None
        gc.collect()
        
        # Rebuild the index with the new documents
        print(f"Building index...")
        self._build_index()
        print(f"Document processing complete")
        
    def simple_chunk(self, text, chunk_size=1024, overlap=50):
        """Split document into chunks with minimal memory usage.
        
        Args:
            text: The document text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        # Strip the text to remove extra whitespace
        text = text.strip()
        
        # Handle empty text
        if not text:
            return []
            
        # Handle text shorter than chunk_size
        if len(text) <= chunk_size:
            return [text]
            
        # Initialize result
        chunks = []
        
        # Calculate effective chunk size (accounting for overlap)
        stride = chunk_size - overlap
        
        # Split text into chunks with overlap
        for i in range(0, len(text), stride):
            # Get chunk of text
            chunk = text[i:i + chunk_size]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
                
            # Free memory after each chunk
            if i % (stride * 10) == 0:
                gc.collect()
                
        return chunks
        
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    # Clear memory after each page
                    if page_num % 10 == 0:
                        gc.collect()
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            text = f"[Error processing document: {pdf_path}]"
        return text
    
    def _build_index(self):
        """Build FAISS index from documents."""
        if not self.documents:
            return
            
        # Generate embeddings for documents in batches to save memory
        batch_size = 64  # Process documents in larger batches
        dimension = 384  # Fixed dimension for the MiniLM-L6-v2 model
        
        # Create a new index - more efficient than updating
        index = faiss.IndexFlatIP(dimension)
        
        # Process in batches
        for i in range(0, len(self.documents), batch_size):
            end_idx = min(i + batch_size, len(self.documents))
            batch = self.documents[i:end_idx]
            
            # Encode batch
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(batch_embeddings)
            
            # Add to index
            index.add(batch_embeddings.astype(np.float32))
            
            # Force garbage collection
            batch_embeddings = None
            gc.collect()
            
        # Replace the old index
        self.index = index
        
        # Force final garbage collection
        gc.collect()

    def retrieve(self, query: str, top_n=5, threshold=0.2):
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
        search_k = min(top_n * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        # Get the documents with their metadata
        results = []
        seen_sources = set()  # To track unique sources
        
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > threshold:  # Apply similarity threshold
                if idx < 0 or idx >= len(self.documents):  # Guard against out-of-bounds indices
                    continue
                    
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
        
        # Clean up
        query_embedding = None
        gc.collect()
        
        if not results:
            return ["No relevant information found for your query. Please try a different question."]
        
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
