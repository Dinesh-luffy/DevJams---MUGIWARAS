# app.py
import os
import uvicorn
import shutil
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from rag import load_pdf_text, chunk_text, store_in_faiss, retrieve_context
from llm_gen import generate_answer

# --- CONFIG ---
BASE_DB_PATH = "data/vector_store/"
embedding_model = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
current_case_db_path = None

app = FastAPI(title="Courtroom AI Assistant", version="1.0")


# -------------------- Utility Functions --------------------
def list_cases():
    if not os.path.exists(BASE_DB_PATH):
        return []
    return [d for d in os.listdir(BASE_DB_PATH) if os.path.isdir(os.path.join(BASE_DB_PATH, d))]


# -------------------- API Routes --------------------

@app.get("/")
def read_root():
    return {"message": "Legal AI Assistant API is running! Go to /docs to try it."}

@app.post("/case/create")
def create_case(case_name: str = Form(...)):
    global current_case_db_path
    case_path = os.path.join(BASE_DB_PATH, case_name)
    os.makedirs(case_path, exist_ok=True)
    current_case_db_path = case_path
    return {"message": f"Case '{case_name}' created", "active_case": case_name}


@app.get("/case/list")
def get_cases():
    cases = list_cases()
    return {"cases": cases}


@app.post("/case/select")
def select_case(case_name: str = Form(...)):
    global current_case_db_path
    case_path = os.path.join(BASE_DB_PATH, case_name)
    if not os.path.exists(case_path):
        return JSONResponse(status_code=404, content={"error": "Case not found"})
    current_case_db_path = case_path
    return {"message": f"Case '{case_name}' selected", "active_case": case_name}


@app.post("/case/upload_pdf")
async def upload_pdf(file: UploadFile):
    global current_case_db_path
    if not current_case_db_path:
        return JSONResponse(status_code=400, content={"error": "Select or create a case first"})

    pdf_path = os.path.join(current_case_db_path, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)
    store_in_faiss(chunks, current_case_db_path)

    return {"message": f"PDF '{file.filename}' uploaded and indexed"}


@app.post("/case/query")
def ask_case_question(query: str = Form(...)):
    global current_case_db_path
    if not current_case_db_path:
        return JSONResponse(status_code=400, content={"error": "Select or create a case first"})

    context = retrieve_context(query, current_case_db_path, top_k=3)
    answer = generate_answer(query, context)
    return {"query": query, "answer": answer}


@app.post("/general/query")
def ask_general_question(query: str = Form(...)):
    answer = generate_answer(query, context="")
    return {"query": query, "answer": answer}


@app.post("/case/opponent")
def opponent_argument(opponent_text: str = Form(...)):
    global current_case_db_path
    if not current_case_db_path:
        return JSONResponse(status_code=400, content={"error": "Select or create a case first"})

    context = retrieve_context(opponent_text, current_case_db_path, top_k=3)
    query = f"The opponent argued: {opponent_text}. Suggest counter points using available legal context."
    answer = generate_answer(query, context)
    return {"opponent": opponent_text, "suggested_response": answer}


# -------------------- Run Server --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
