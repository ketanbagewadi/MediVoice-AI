# ============================================================
# main.py — FastAPI Backend
# ============================================================
# Two endpoints:
#   POST /transcribe → audio file → transcribed text
#   POST /analyze    → patient text → medical suggestions
#   GET  /history    → returns conversation history
#   DELETE /history  → clears history
#
# Run with:  uvicorn main:app --reload --port 8000
# ============================================================

import os
import uuid
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our STT and LLM modules
from stt import transcribe_audio
from llm import analyze_symptoms

# ----------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------
app = FastAPI(
    title="Medical Voice Assistant API",
    description="Speech-to-text + BioMistral medical analysis",
    version="1.0.0"
)

# Allow requests from the HTML frontend (CORS)
# In production, replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp folder to store uploaded audio files
UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory conversation history (resets when server restarts)
# Format: list of {"role": "user"/"assistant", "text": "...", "time": "..."}
conversation_history: List[dict] = []


# ----------------------------------------------------------
# SCHEMAS
# ----------------------------------------------------------
class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint"""
    text: str  # the transcribed patient text


class HistoryItem(BaseModel):
    """One item in conversation history"""
    role: str       # "user" or "assistant"
    text: str
    time: str


# ----------------------------------------------------------
# ROUTES
# ----------------------------------------------------------

@app.get("/")
def root():
    """Health check — visit http://localhost:8000 to confirm it's running"""
    return {"status": "running", "message": "Medical Voice Assistant API is live!"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accepts an audio file, saves it temporarily,
    runs Wav2Vec2 transcription, and returns the text.

    curl example:
      curl -X POST http://localhost:8000/transcribe \
           -F "audio=@patient_voice.wav"
    """

    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4"]
    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {audio.content_type}. Use wav, mp3, ogg, or webm."
        )

    # Save uploaded file with a unique name to avoid conflicts
    filename  = f"{uuid.uuid4()}_{audio.filename}"
    filepath  = os.path.join(UPLOAD_DIR, filename)

    try:
        # Write the uploaded bytes to disk
        with open(filepath, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Run Wav2Vec2 transcription
        transcription = transcribe_audio(filepath)

        # Save to history
        conversation_history.append({
            "role": "user",
            "text": transcription,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return {
            "success": True,
            "transcription": transcription
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Always clean up the temp file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Accepts transcribed patient text,
    sends it to BioMistral, and returns medical suggestions.

    curl example:
      curl -X POST http://localhost:8000/analyze \
           -H "Content-Type: application/json" \
           -d '{"text": "I have a headache and fever"}'
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        suggestions = analyze_symptoms(request.text)

        # Save assistant response to history
        conversation_history.append({
            "role": "assistant",
            "text": suggestions,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return {
            "success": True,
            "suggestions": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/history")
def get_history():
    """Returns the full conversation history for this session."""
    return {
        "success": True,
        "count": len(conversation_history),
        "history": conversation_history
    }


@app.delete("/history")
def clear_history():
    """Clears the conversation history."""
    conversation_history.clear()
    return {"success": True, "message": "History cleared."}