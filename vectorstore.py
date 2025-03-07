import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import PyPDF2

# Initialize empty storage for documents and their text
class VectorStore:
    def __init__(self):
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

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
            
        # Tokenize the document
        tokens = word_tokenize(doc_text.lower())
        # Remove punctuation and stopwords (optional)
        tokens = [word for word in tokens if word not in string.punctuation]
        self.documents.append(doc_text)
        self.tokenized_docs.append(tokens)
        
        # Rebuild BM25 index (efficiently done only after bulk insertions)
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            text = f"[Error processing document: {pdf_path}]"
        return text

    def retrieve(self, query: str, top_n=1):
        """Retrieve top_n relevant documents based on BM25 ranking."""
        if not self.bm25 or not self.documents:
            return ["No documents available for retrieval."]
            
        query_tokens = word_tokenize(query.lower())
        query_tokens = [word for word in query_tokens if word not in string.punctuation]
        
        # Score the query against the stored documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get indices of top_n most relevant documents
        top_docs_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        # Return the corresponding document texts
        return [self.documents[idx] for idx in top_docs_idx]

# Example usage (store this in your server code)
vector_store = VectorStore()

# Add a few documents
vector_store.add_document("The quick brown fox jumps over the lazy dog.")
vector_store.add_document("Artificial intelligence is the future of technology.")
vector_store.add_document("The fox is known for its cunning and agility.")

# Query retrieval
query = "Tell me about the fox"
results = vector_store.retrieve(query, top_n=1)

print("Top matching documents:")
for result in results:
    print(result)
