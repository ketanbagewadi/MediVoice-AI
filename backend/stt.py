# ============================================================
# stt.py — Speech-to-Text using Wav2Vec2
# ============================================================
# What this file does:
#   1. Loads the Wav2Vec2 model from Hugging Face (once, at startup)
#   2. Accepts an audio file (wav/mp3/etc.)
#   3. Converts the audio waveform → text using CTC decoding
# ============================================================

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ----------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------
# We use facebook/wav2vec2-base-960h — a pretrained English ASR model.
# It's ~360MB and works on CPU (slow but works without a GPU).
# "processor" handles: audio normalization + tokenization
# "model"     handles: the actual speech recognition

MODEL_NAME = "facebook/wav2vec2-base-960h"

print("[STT] Loading Wav2Vec2 model... (only once at startup)")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()  # set to evaluation/inference mode (no training)
print("[STT] Wav2Vec2 model loaded successfully!")


# ----------------------------------------------------------
# MAIN FUNCTION: transcribe_audio
# ----------------------------------------------------------
def transcribe_audio(audio_path: str) -> str:
    """
    Takes a path to an audio file and returns the transcribed text.

    Steps:
      1. Load audio file and resample to 16000 Hz (Wav2Vec2 requirement)
      2. Convert waveform to model input tensors
      3. Run the model to get logits (raw predictions)
      4. Decode logits → readable text using CTC
    
    Args:
        audio_path: Path to the audio file (wav, mp3, ogg, etc.)
    
    Returns:
        Transcribed text as a string
    """

    # Step 1: Load audio
    # librosa loads any audio format and resamples it to 16000 Hz
    # Wav2Vec2 was trained on 16kHz audio, so this is mandatory
    print(f"[STT] Loading audio from: {audio_path}")
    speech_array, sample_rate = librosa.load(audio_path, sr=16000)
    # speech_array → numpy array of float32 values (the raw waveform)
    # sample_rate  → will be 16000 because we forced sr=16000

    # Step 2: Preprocess — normalize the audio + convert to PyTorch tensor
    inputs = processor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",   # pt = PyTorch tensors
        padding=True
    )
    # inputs.input_values → shape: [1, num_audio_samples]

    # Step 3: Run through the model (no gradient needed for inference)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    # logits → shape: [1, time_steps, vocab_size]
    # Each time step has scores for every possible character/token

    # Step 4: CTC Decoding — pick the most likely character at each step
    predicted_ids = torch.argmax(logits, dim=-1)
    # predicted_ids → shape: [1, time_steps], integer token IDs

    # Convert token IDs → actual text string
    transcription = processor.decode(predicted_ids[0])
    # .decode() maps IDs back to characters and removes CTC blank tokens

    print(f"[STT] Transcription: {transcription}")
    return transcription.strip()


# ----------------------------------------------------------
# QUICK TEST (run this file directly to test)
# ----------------------------------------------------------
# Usage: python stt.py
# It will try to transcribe a file called "test.wav" in current dir
if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    result = transcribe_audio(test_file)
    print(f"\nFinal Transcription:\n{result}")