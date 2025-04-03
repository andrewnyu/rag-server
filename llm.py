import requests
from vectorstore import MultiFileRAG
#from langchain_core.prompts import PromptTemplate

class LLM:
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url

    def generate(self, query, docs):
        """ Send request to remote LLM inference API with improved context handling """
        # Format context with better separation
        if docs and isinstance(docs, list):
            # The documents are already formatted by the vectorstore
            context = "\n\n---\n\n".join(docs)
        else:
            context = "No relevant information found."
            
        # Enhanced prompt format with better instructions
        prompt = f"""Answer the following question based ONLY on the provided context. 
        If the context doesn't contain the information needed to answer the question, 
        say "I don't have enough information to answer this question" instead of making up an answer.

Context:
{context}

Question: {query}

Instructions:
1. Use ONLY the information from the provided context
2. If the answer is not in the context, admit you don't know
3. Provide a concise and accurate answer
4. Cite the relevant document numbers when possible (e.g., "According to Document 1...")

Answer:"""
        
        response = requests.get(self.endpoint_url + prompt)

        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code}, {response.text}"
