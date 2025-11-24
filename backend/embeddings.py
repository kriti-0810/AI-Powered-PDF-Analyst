from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Loads the SentenceTransformer embedding model.
        """
        print("🔍 Loading embedding model... (first time may take ~20 seconds)")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, texts):
        """
        Takes a list of text chunks.
        Returns a list of embeddings (vectors).
        """
        return self.model.encode(texts, convert_to_numpy=True)
