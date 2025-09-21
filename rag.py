# rag.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

def load_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split text into chunks for embeddings"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def store_in_faiss(chunks: list[str], db_path: str):
    """Store text chunks in a specific FAISS DB (append if exists)"""
    # Correctly checks for the index file itself to avoid errors
    if os.path.exists(os.path.join(db_path, "index.faiss")):
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    else:
        # Create a new FAISS index if the files don't exist
        db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(db_path)
    print(f"✅ Stored {len(chunks)} chunks in FAISS DB at {db_path}.")

def retrieve_context(query: str, db_path: str, top_k: int = 3) -> str:
    """Search in a specific FAISS DB and return concatenated context"""
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        print("⚠️ No FAISS index found for this case. Upload PDFs first!")
        return ""
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)
    context = "\n\n".join([res.page_content for res in results])
    return context