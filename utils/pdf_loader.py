import os
from PyPDF2 import PdfReader

def load_pdfs(pdf_dir="input_pdfs"):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append({"filename": filename, "content": text})
    return documents
