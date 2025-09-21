import os
import json
from ollama import Client

# --- CONFIG ---
CASE_TXT_FOLDER = "cases_raw"  # folder with your parsed case text files
OUTPUT_FOLDER = "cases_qa_txt" # folder to store Q&A text files
MODEL_NAME = "llama2:latest"
NUM_QA_PAIRS = 6

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_qa(case_text, num_pairs=NUM_QA_PAIRS):
    """
    Generate Q&A pairs from a case text using LLaMA2 via Ollama.
    Returns a string in plain Q: ... A: ... format.
    """
    client = Client()

    prompt = f"""
You are a legal AI assistant. From the court judgment text below, generate {num_pairs} useful question-answer pairs.

⚠️ IMPORTANT:
- ONLY output Q&A pairs.
- Format exactly:
  Q: [question]
  A: [answer]
- DO NOT summarize, do not add headings, do not use bullets or numbers.

EXAMPLES:
Q: What was the main issue in the case?
A: The dispute concerned whether the Central Government could set qualifications for Motor Vehicle Inspector in J&K.

Q: What did the Supreme Court decide about the State's power?
A: The State has the authority to prescribe qualifications for Motor Vehicle Inspector posts.

Q: What was the Court's view on the corrigendum?
A: The corrigendum including minimum qualifications by the Central Government was valid and not invalidated.

Now generate {num_pairs} new Q&A pairs based ONLY on the judgment text below.

Court Judgment Text:
{case_text}
"""

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # ✅ FIX: Access the content correctly from the response dictionary
        qa_text = response['message']['content']
        return qa_text

    except Exception as e:
        print(f"❌ An unexpected error occurred during generation: {e}")
        return ""


def main():
    case_files = [f for f in os.listdir(CASE_TXT_FOLDER) if f.endswith(".txt")]

    if not case_files:
        print("⚠️ No case text files found in the folder.")
        return

    for case_file in case_files:
        print(f"\nGenerating Q&A for: {case_file} ...")
        case_path = os.path.join(CASE_TXT_FOLDER, case_file)

        # Read case text
        with open(case_path, "r", encoding="utf-8") as f:
            case_text = f.read()

        qa_text = generate_qa(case_text, NUM_QA_PAIRS)

        if qa_text:
            txt_filename = os.path.splitext(case_file)[0] + "_qa.txt"
            txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(qa_text.strip())
            print(f"✅ Saved Q&A to {txt_path}")
        else:
            print(f"❌ No Q&A pairs generated for {case_file}")


if __name__ == "__main__":
    main()