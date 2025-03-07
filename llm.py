import requests

class LLM:
    def __init__(self, endpoint_url="http://your-gpu-endpoint-url/query?text="):
        self.endpoint_url = endpoint_url

    def generate(self, query, docs):
        """ Send request to remote LLM inference API """
        prompt = f"Context: {' '.join(docs)}\nQuestion: {query}\nAnswer:"
        response = requests.get(self.endpoint_url + prompt)

        if response.status_code == 200:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code}, {response.text}"
