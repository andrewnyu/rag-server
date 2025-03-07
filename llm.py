import requests

class LLM:
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url

    def generate(self, query, docs):
        """ Send request to remote LLM inference API """
        # Create a prompt that instructs the LLM to only provide the answer without repeating the context
        prompt = f"Context: {' '.join(docs)}\nQuestion: {query}\nInstructions: Provide a direct answer to the question based on the context. Do not repeat or reference the context in your answer.\nAnswer:"
        response = requests.get(self.endpoint_url + prompt)

        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code}, {response.text}"
