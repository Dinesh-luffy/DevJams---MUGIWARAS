import os
import re
import csv

# --- CONFIG ---
INPUT_FOLDER = "data1/parsed"  # folder containing messy Q&A txt files
OUTPUT_CSV = "all_cases_qa.csv"

def parse_numbered_qa(text):
    """
    Convert numbered Q&A text into a list of dictionaries.
    """
    qa_pairs = []
    # Split by numbered questions like 1., 2., 3., etc.
    blocks = re.split(r'\n?\d+\.\s', text)
    for block in blocks[1:]:  # skip first empty split
        parts = block.strip().split('\n', 1)
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

def save_to_csv(qa_pairs, csv_path):
    """
    Save list of Q&A dictionaries to a CSV file.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for pair in qa_pairs:
            writer.writerow({"question": pair["question"], "answer": pair["answer"]})
    print(f"✅ Saved {len(qa_pairs)} Q&A pairs to {csv_path}")

def main():
    all_qa = []

    # Loop through all txt files in the input folder
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".txt"):
            file_path = os.path.join(INPUT_FOLDER, file_name)
            print(f"Processing {file_name} ...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            qa_pairs = parse_numbered_qa(text)
            all_qa.extend(qa_pairs)
    
    # Save everything into one CSV
    save_to_csv(all_qa, OUTPUT_CSV)
    print(f"✅ Total Q&A pairs processed: {len(all_qa)}")

if __name__ == "__main__":
    main()
