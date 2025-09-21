import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

# Path to FAISS index storage
DB_FAISS_PATH = "data/vector_store/faiss_index"

# Use LEGAL-BERT embeddings for better legal understanding
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")


def load_pdf_text(pdf_path):
    """Extract raw text from a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


def store_in_faiss(chunks):
    """Store text chunks into FAISS index (append if exists)."""
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    else:
        db = FAISS.from_texts(chunks, embedding_model)

    db.save_local(DB_FAISS_PATH)
    print(f"✅ Stored {len(chunks)} chunks in FAISS DB.")


def search_query(query, top_k=3):
    """Search FAISS DB for most relevant text chunks."""
    if not os.path.exists(DB_FAISS_PATH):
        print("⚠️ No FAISS index found. Please upload PDFs first!")
        return []

    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)

    print(f"\n🔎 Top {top_k} results for query: {query}\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:300]}...\n")

    return results


def main():
    while True:
        print("\nOptions:")
        print("1. Upload a new PDF")
        print("2. Ask a legal question")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")

        if choice == "1":
            pdf_path = input("Enter full path to PDF: ").strip()
            if os.path.exists(pdf_path):
                text = load_pdf_text(pdf_path)
                chunks = chunk_text(text)
                store_in_faiss(chunks)
            else:
                print("❌ File not found.")

        elif choice == "2":
            query = input("Enter your legal question: ").strip()
            search_query(query, top_k=3)

        elif choice == "3":
            print("👋 Exiting...")
            break
        else:
            print("⚠️ Invalid choice, try again.")


if __name__ == "__main__":
    main()
