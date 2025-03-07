from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        return np.array(self.model.encode(text), dtype=np.float32)

    def dim(self):
        return self.model.get_sentence_embedding_dimension()
