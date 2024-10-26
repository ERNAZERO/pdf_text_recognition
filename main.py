from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from model_utils import initialize_model_and_db, load_and_split_pdf, create_vector_db, answer_question_from_pdf
import shutil
import os

app = FastAPI()

# Инициализация модели при запуске сервера
llm = initialize_model_and_db()
vector_db = None  # Инициализация переменной для базы данных

@app.post("/ask-pdf")
async def ask_pdf(pdf_file: UploadFile, question: str = Form(...)):
    global vector_db

    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Загрузите файл в формате PDF.")

    temp_pdf_path = f"docs/{pdf_file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)

    chunks = load_and_split_pdf(temp_pdf_path)

    if not chunks:
        os.remove(temp_pdf_path)
        return JSONResponse(content={"error": "Leere PDF-Datei"}, status_code=400)

    # Создаем векторную базу данных
    vector_db = create_vector_db(chunks)

    answer = answer_question_from_pdf(vector_db, llm, question)

    response = JSONResponse(content={"answer": answer})

    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    vector_db = None

    return response

