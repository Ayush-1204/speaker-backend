# app/main.py
import torchaudio
torchaudio.set_audio_backend("soundfile")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
import shutil
import numpy as np

from app.utils.feature_extractor import get_ecapa_embedding_from_file
from app.utils.storage import save_embedding, load_all_embeddings
from app.utils.compare import match_embedding, THRESHOLD

app = FastAPI()

TEMP_DIR = "app/data/temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)


# ------------------ TEST CONNECTION ------------------
@app.get("/test_connection")
def test_connection():
    return {"status": "ok"}


# ------------------ LIST SPEAKERS ------------------
@app.get("/list_speakers")
def list_speakers():
    users = load_all_embeddings()
    return {"speakers": list(users.keys())}

#---------------------DELETE----------------------
@app.delete("/delete_speaker/{name}")
def delete_speaker(name: str):
    speaker_dir = os.path.join("app", "data", "familiar_embeddings", name)

    print("🔍 Trying to delete:", speaker_dir)

    if not os.path.exists(speaker_dir):
        raise HTTPException(status_code=404, detail="Speaker not found")

    import shutil
    shutil.rmtree(speaker_dir)

    return {"status": "deleted", "speaker": name}


# ------------------ CLEAR TEMP AUDIO ------------------
@app.post("/clear_temp_audio")
def clear_temp_audio():
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        return {"status": "success", "message": "Temporary audio files cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear temporary audio files: {e}")


# ------------------ ENROLL ------------------
@app.post("/enroll")
async def enroll(
    name: str = Form(...),              # <-- FIX for Android + multipart
    file: UploadFile = File(...)
):
    temp_path = os.path.join(TEMP_DIR, file.filename)

    # Save temp audio file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract ECAPA embedding
    emb = get_ecapa_embedding_from_file(temp_path)

    # Store embedding
    save_embedding(name, emb)

    return {
        "status": "enrolled",
        "name": name,
        "embedding_dim": len(emb)
    }


# ------------------ VERIFY (FAMILIARITY CHECK) ------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):

    temp_path = os.path.join(TEMP_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract embedding
    query_emb = get_ecapa_embedding_from_file(temp_path)

    # Load all stored embeddings
    users = load_all_embeddings()

    best_match = None
    best_score = 0.0

    for user, stored_emb in users.items():
        score = match_embedding(query_emb, stored_emb)
        if score > best_score:
            best_score = score
            best_match = user

    # Thresholding
    if best_score >= THRESHOLD:
        return {
            "result": "familiar",
            "name": best_match,
            "similarity": float(best_score)
        }

    return {
        "result": "stranger",
        "similarity": float(best_score)
    }


# ------------------ ANDROID USES /recognize ------------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    return await verify(file)


# ------------------ ROOT ROUTE ------------------
@app.get("/")
def home():

    return {"message": "Speaker Recognition Backend Running"}
