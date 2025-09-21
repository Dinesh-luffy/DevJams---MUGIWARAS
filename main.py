import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm_gen import generate_answer  # yo

DB_FAISS_PATH = "data/vector_store/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

def load_pdf_text(pdf_path):
    """Extract text from PDF"""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def store_in_faiss(chunks):
    """Store text chunks in FAISS DB (append if exists)"""
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    else:
        db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Stored {len(chunks)} chunks in FAISS DB.")

def retrieve_context(query, top_k=3):
    """Search in FAISS DB and return concatenated context"""
    if not os.path.exists(DB_FAISS_PATH):
        return ""
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)
    context = "\n\n".join([res.page_content for res in results])
    return context

# ---------------- MAIN LOOP ----------------
def main():
    while True:
        print("\nOptions:")
        print("1. Upload a new PDF")
        print("2. Ask a legal question")
        print("3. Opponent says something (simulate live argument)")
        print("4. Exit")
        choice = input("Enter choice (1/2/3/4): ").strip()

        if choice == "1":
            pdf_path = input("Enter full path to PDF: ").strip()
            if os.path.exists(pdf_path):
                text = load_pdf_text(pdf_path)
                chunks = chunk_text(text)
                store_in_faiss(chunks)
            else:
                print("⚠️ File not found.")
        elif choice == "2":
            query = input("Enter your legal question: ").strip()
            context = retrieve_context(query, top_k=3)
            answer = generate_answer(query, context)
            print("\n💡 Answer:\n")
            print(answer)


        elif choice == "3":
            opponent_text = input("Enter what the opponent lawyer said: ").strip()
            context = retrieve_context(opponent_text, top_k=3)
            query = f"The opponent argued: {opponent_text}. Suggest counter points using available legal context."
            answer = generate_answer(query, context)
        
            print("\n💡 Suggested Response:\n")
            print(answer)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("⚠️ Invalid choice, try again.")

if __name__ == "__main__":
    main()
