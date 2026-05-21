# ============================================================
# llm.py — Medical Analysis using BioMistral-7B
# ============================================================
# What this file does:
#   1. Loads BioMistral-7B from Hugging Face (once, at startup)
#   2. Takes transcribed patient text as input
#   3. Builds a structured medical prompt
#   4. Returns medical suggestions
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "BioMistral/BioMistral-7B"

# Auto-detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[LLM] Using device: {device}")
print("[LLM] Loading BioMistral-7B... (first time download may take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",       # auto-place layers on GPU/CPU
    low_cpu_mem_usage=True,  # saves RAM during loading
)
model.eval()
print("[LLM] BioMistral-7B ready!")


def build_prompt(patient_text: str) -> str:
    """
    Wraps patient text in a Mistral-style [INST] prompt with
    a medical system instruction so BioMistral knows its role.
    """
    system_instruction = (
        "You are a helpful medical AI assistant. "
        "A patient has described their symptoms. Provide:\n"
        "1. Possible conditions or causes\n"
        "2. Suggested OTC medicines or home remedies\n"
        "3. Important precautions\n"
        "4. When to see a doctor\n"
        "Be clear, structured, and remind the patient to consult a real doctor."
    )
    return f"[INST] {system_instruction}\n\nPatient says: \"{patient_text}\" [/INST]"


def analyze_symptoms(patient_text: str, max_new_tokens: int = 512) -> str:
    """
    Takes patient text → returns structured medical suggestions.

    Steps:
      1. Build a structured medical prompt
      2. Tokenize the prompt into IDs
      3. Generate tokens with the model
      4. Decode only the NEW tokens (model's answer, not the prompt)
    """
    prompt = build_prompt(patient_text)
    print(f"[LLM] Analyzing: {patient_text[:80]}...")

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,       # creativity level
            top_p=0.9,             # nucleus sampling
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # Decode only the response (strip the input prompt tokens)
    input_length = inputs["input_ids"].shape[1]
    new_tokens   = output_ids[0][input_length:]
    response     = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"[LLM] Done. Response length: {len(response)} chars")
    return response.strip()


# Quick test
if __name__ == "__main__":
    test = "I have a headache, slight fever, and sore throat for 2 days."
    print(analyze_symptoms(test))