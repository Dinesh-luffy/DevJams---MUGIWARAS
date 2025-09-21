import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Legal-BERT embeddings model
model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

# Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Build or update FAISS index
def build_faiss(documents, index_path="data/faiss_index/index.faiss", vectors_path="data/faiss_index/vectors.npy", meta_path="data/faiss_index/metadata.npy"):
    os.makedirs("data/faiss_index", exist_ok=True)

    # Prepare text chunks
    texts, metadata = [], []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadata.append(doc["filename"])

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    # If FAISS exists, load and extend
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        old_embeddings = np.load(vectors_path)
        old_metadata = list(np.load(meta_path, allow_pickle=True))
        new_embeddings = np.vstack((old_embeddings, embeddings))
        new_metadata = old_metadata + metadata
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        new_embeddings = embeddings
        new_metadata = metadata

    # Add vectors
    index.add(new_embeddings)
    faiss.write_index(index, index_path)
    np.save(vectors_path, new_embeddings)
    np.save(meta_path, np.array(new_metadata, dtype=object))
    print("✅ FAISS index updated with new documents")
