import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path to store FAISS index
DB_FAISS_PATH = "data/vector_store/faiss_index"

# Use Legal-BERT embeddings (Hugging Face model)
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

def store_in_faiss(chunks: list[str]):
    """Store text chunks in FAISS DB (append if exists)"""
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    else:
        db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Stored {len(chunks)} chunks in FAISS DB.")

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Search in FAISS DB and return concatenated context"""
    if not os.path.exists(DB_FAISS_PATH):
        print("⚠️ No FAISS index found. Upload PDFs first!")
        return ""
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)
    context = "\n\n".join([res.page_content for res in results])
    return context
