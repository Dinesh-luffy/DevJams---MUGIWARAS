import os
from PyPDF2 import PdfReader

INPUT_DIR = "cases"
OUTPUT_DIR = "data1/parsed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            print(f"⚠️ Page {i+1} of {pdf_path} is empty (maybe scanned).")
        text += page_text or ""
    return text

def save_text(filename, text):
    output_path = os.path.join(OUTPUT_DIR, filename + ".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved {output_path}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input folder not found: {INPUT_DIR}")
    else:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
        if not files:
            print(f"⚠️ No PDF files found in {INPUT_DIR}")
        else:
            for file in files:
                pdf_path = os.path.join(INPUT_DIR, file)
                print(f"📂 Processing {pdf_path} ...")
                text = parse_pdf(pdf_path)

                if text.strip():
                    save_text(file.replace(".pdf", ""), text)
                else:
                    print(f"❌ No text extracted from {file} (probably scanned PDF).")
