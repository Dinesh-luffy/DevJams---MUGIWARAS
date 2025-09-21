# main.py
import os
import shutil
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm_gen import generate_answer
from rag import load_pdf_text, chunk_text, store_in_faiss, retrieve_context

# Base directory to store all case databases
BASE_DB_PATH = "data/vector_store/"
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")

# Global variable to hold the current case's database path
current_case_db_path = None

def get_voice_input():
    """
    Listens to the microphone and converts the speech to text.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"✅ Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"⚠️ Could not request results from Google Speech Recognition service; {e}")
        return None

def list_cases():
    """Lists all available case directories."""
    if not os.path.exists(BASE_DB_PATH):
        print("⚠️ No cases found.")
        return []
    cases = [d for d in os.listdir(BASE_DB_PATH) if os.path.isdir(os.path.join(BASE_DB_PATH, d))]
    if not cases:
        print("⚠️ No cases found.")
    else:
        print("\n📂 Available Cases:")
        for i, case in enumerate(cases):
            print(f"  {i+1}. {case}")
    return cases

# ---------------- MAIN LOOP ----------------
def main():
    global current_case_db_path
    while True:
        if current_case_db_path:
            case_name = os.path.basename(current_case_db_path)
            print(f"\nActive Case: {case_name}")
        else:
            print("\nActive Case: None")
        
        print("\nOptions:")
        print("1. Create New Case")
        print("2. Select Existing Case")
        print("3. Upload a new PDF to active case")
        print("4. Ask a question about active case")
        print("5. Ask a general legal question")
        print("6. Opponent says something (simulate live argument)")
        print("7. Exit")
        choice = input("Enter choice (1/2/3/4/5/6/7): ").strip()

        if choice == "1":
            case_name = input("Enter new case name: ").strip()
            if case_name:
                case_path = os.path.join(BASE_DB_PATH, case_name)
                os.makedirs(case_path, exist_ok=True)
                current_case_db_path = case_path
                print(f"✅ Case '{case_name}' created and set as active.")
            else:
                print("⚠️ Case name cannot be empty.")
        elif choice == "2":
            cases = list_cases()
            if cases:
                try:
                    case_index = int(input("Enter case number to select: ").strip()) - 1
                    if 0 <= case_index < len(cases):
                        case_name = cases[case_index]
                        current_case_db_path = os.path.join(BASE_DB_PATH, case_name)
                        print(f"✅ Case '{case_name}' set as active.")
                    else:
                        print("⚠️ Invalid case number.")
                except (ValueError, IndexError):
                    print("⚠️ Invalid input. Please enter a number from the list.")
        elif choice == "3":
            if not current_case_db_path:
                print("⚠️ Please select or create a case first.")
                continue
            path_input = input("Enter full path to PDF or 'voice' for voice input: ").strip()
            if path_input.lower() == "voice":
                pdf_path = get_voice_input()
            else:
                pdf_path = path_input

            if pdf_path and os.path.exists(pdf_path):
                text = load_pdf_text(pdf_path)
                chunks = chunk_text(text)
                store_in_faiss(chunks, current_case_db_path)
            else:
                print("⚠️ File not found or voice input failed.")
        elif choice == "4":
            if not current_case_db_path:
                print("⚠️ Please select or create a case first.")
                continue
            query_input = input("Enter your legal question or 'voice' for voice input: ").strip()
            if query_input.lower() == "voice":
                query = get_voice_input()
            else:
                query = query_input

            if query:
                context = retrieve_context(query, current_case_db_path, top_k=3)
                answer = generate_answer(query, context)
                print("\n💡 Answer:\n")
                print(answer)
        elif choice == "5":
            query_input = input("Enter your general legal question or 'voice' for voice input: ").strip()
            if query_input.lower() == "voice":
                query = get_voice_input()
            else:
                query = query_input

            if query:
                # Bypass RAG and pass an empty context
                answer = generate_answer(query, context="")
                print("\n💡 Answer:\n")
                print(answer)
        elif choice == "6":
            if not current_case_db_path:
                print("⚠️ Please select or create a case first.")
                continue
            opponent_input = input("Enter what the opponent lawyer said or 'voice' for voice input: ").strip()
            if opponent_input.lower() == "voice":
                opponent_text = get_voice_input()
            else:
                opponent_text = opponent_input

            if opponent_text:
                context = retrieve_context(opponent_text, current_case_db_path, top_k=3)
                query = f"The opponent argued: {opponent_text}. Suggest counter points using available legal context."
                answer = generate_answer(query, context)
                print("\n💡 Suggested Response:\n")
                print(answer)
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("⚠️ Invalid choice, try again.")

if __name__ == "__main__":
    main()