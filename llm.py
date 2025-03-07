import requests

class LLM:
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url

    def generate(self, query, docs):
        """ Send request to remote LLM inference API """
        # Format context with better separation
        if docs and isinstance(docs, list):
            # Format each document with a separator
            formatted_docs = []
            for i, doc in enumerate(docs):
                formatted_docs.append(f"Document {i+1}:\n{doc}")
            
            # Join all formatted documents
            context = "\n\n---\n\n".join(formatted_docs)
        else:
            context = "No relevant information found."
            
        # Effective prompt format
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        response = requests.get(self.endpoint_url + prompt)

        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code}, {response.text}"
