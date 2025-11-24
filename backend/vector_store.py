import os
import faiss
import numpy as np
import json

FAISS_INDEX_PATH = "data/faiss_index/index.faiss"
METADATA_PATH = "data/faiss_index/metadata.json"

class FAISSStore:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []

    def create_new_index(self):
        """Create a new FAISS index (L2 distance)."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings, chunk_metadata):
        """
        embeddings: numpy array of shape (N, 384)
        chunk_metadata: list of dicts, each describing a chunk
        """
        if self.index is None:
            self.create_new_index()

        self.index.add(embeddings)
        self.metadata.extend(chunk_metadata)

    def search(self, query_embedding, top_k=5):
        """
        Returns the top K most relevant chunks.
        """
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])
        return results

    def save_index(self):
        """Save FAISS index + metadata to disk."""
        if self.index is None:
            return

        faiss.write_index(self.index, FAISS_INDEX_PATH)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load_index(self):
        """Load FAISS index + metadata."""
        if not os.path.exists(FAISS_INDEX_PATH):
            return False

        self.index = faiss.read_index(FAISS_INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        return True
