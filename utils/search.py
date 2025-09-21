import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

def search(query, index_path="data/faiss_index/index.faiss", vectors_path="data/faiss_index/vectors.npy", meta_path="data/faiss_index/metadata.npy", top_k=3):
    # Load FAISS index
    index = faiss.read_index(index_path)
    embeddings = np.load(vectors_path)
    metadata = list(np.load(meta_path, allow_pickle=True))

    # Encode query
    query_vector = model.encode([query], convert_to_numpy=True)

    # Search nearest neighbors
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            results.append((metadata[idx], embeddings[idx], dist))
    return results
