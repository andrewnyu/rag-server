import requests

class LLM:
    def __init__(self, endpoint_url=None):
        self.endpoint_url = endpoint_url

    def expand_query(self, query):
        expansion_prompt = f"""You are an expert Home Depot product advisor. Expand the following customer query with relevant synonyms and key technical details for better retrieval:

Original Query: {query}

Expanded Query:"""

        try:
            response = requests.get(self.endpoint_url + expansion_prompt)
            if response.status_code == 200:
                return response.json().get("response", query)
            else:
                return query
        except Exception:
            return query

    def generate(self, query, docs):
        if not self.endpoint_url:
            return "Error: No LLM endpoint set. Please configure an endpoint URL."

        expanded_query = self.expand_query(query)

        context = "\n\n---\n\n".join(docs) if docs else "No relevant information found."

        prompt = f"""You are an expert Home Depot engineer providing product recommendations based ONLY on the context provided below. If information is missing, respond with "I don't have enough information to answer this question."

Context:
{context}

Customer Query: {expanded_query}

Instructions:
1. Recommend the best product(s) from the provided context.
2. Briefly explain the choice clearly based on technical suitability.
3. Cite the relevant document numbers when applicable (e.g., "According to Document 1...").

Answer:"""

        try:
            response = requests.get(self.endpoint_url + prompt)
            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"Error: {response.status_code}, {response.text}"
        except Exception as e:
            return f"Error connecting to LLM endpoint: {str(e)}"
