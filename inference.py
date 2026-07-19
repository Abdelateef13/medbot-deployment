"""
inference.py — loads the MedBot model and generates answers.

Design notes (important for the live presentation):

1. The deployed model = frozen base model (EleutherAI/gpt-neox-20b, 4-bit NF4)
   + the LoRA adapter trained in the Colab notebook. We do NOT merge the
   adapter: PEFT loads it on top of the quantized base at runtime, exactly
   like cell #13-14 of the notebook.

2. The prompt template here is copied VERBATIM from the training notebook
   (cell #11). If the inference prompt differs from the training prompt,
   quality collapses — the model was taught to complete this exact format.

3. Generation settings match the notebook's ask_medbot() (cell #20):
   temperature 0.7, top_p 0.9, repetition_penalty 1.15, EOS as pad token.

4. A rule-based emergency safety layer (proposed in the notebook's
   reflection, section #27) runs BEFORE the model: emergency keywords
   always trigger a clear "seek immediate care" banner, so safety does
   not depend on the model remembering to say it.

5. MOCK_MODE=1 lets the Flask app + Docker image be tested on any laptop
   (no GPU, no 12GB download). The container is identical; only the
   backend answer source changes. This is how the UI and Dockerfile were
   verified locally before renting GPU hardware.
"""

import os
import threading

# ----------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ----------------------------------------------------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "EleutherAI/gpt-neox-20b")
# HF Hub repo that contains the trained LoRA adapter
# (adapter_model.safetensors + adapter_config.json + tokenizer files,
#  i.e. the contents of the notebook's OUTPUT_DIR on Google Drive).
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "YOUR_HF_USERNAME/medbot-lora")
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "180"))

# Same system prompt as training (notebook cell #11) — must not change.
SYSTEM_PROMPT = """You are MedBot, a medical information assistant.
Answer the question in simple patient-friendly language.
Do not claim to diagnose the user.
If symptoms may be serious, recommend speaking to a healthcare professional.""".strip()


def make_prompt(question: str) -> str:
    """Identical template to the one used during fine-tuning."""
    return f"""### Instruction:
{SYSTEM_PROMPT}

### Question:
{question}

### Answer:
"""


# ----------------------------------------------------------------------
# Rule-based emergency safety layer (from reflection, section #27)
# ----------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "trouble breathing", "difficulty breathing", "shortness of breath",
    "unconscious", "not breathing", "severe bleeding", "suicide",
    "overdose", "seizure", "anaphylaxis", "choking",
]

EMERGENCY_MESSAGE = (
    "⚠️ Your question mentions symptoms that can be a medical emergency. "
    "If this is happening right now, call your local emergency number "
    "immediately — do not wait for an online answer."
)


def check_emergency(question: str) -> str | None:
    q = question.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in q:
            return EMERGENCY_MESSAGE
    return None


# ----------------------------------------------------------------------
# Model loading — lazy + thread-safe
# ----------------------------------------------------------------------
_model = None
_tokenizer = None
_load_lock = threading.Lock()
_load_error: str | None = None


def model_status() -> dict:
    if MOCK_MODE:
        return {"mode": "mock", "loaded": True, "error": None}
    return {
        "mode": "full",
        "loaded": _model is not None,
        "error": _load_error,
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_REPO,
    }


def load_model():
    """Load the 4-bit base model and attach the LoRA adapter.

    Called once, lazily, on the first request (or from a warmup thread),
    so the container starts fast and the health check responds while the
    ~12GB of weights download.
    """
    global _model, _tokenizer, _load_error
    if MOCK_MODE or _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import PeftModel

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "No CUDA GPU found. bitsandbytes 4-bit quantization "
                    "requires a GPU. Run with MOCK_MODE=1 for a CPU demo, "
                    "or deploy on GPU hardware (e.g. HF Spaces T4)."
                )

            # Same quantization config as training (notebook cell #13)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, ADAPTER_REPO)
            model.eval()
            model.config.use_cache = True  # KV cache back on for fast generation

            _tokenizer = tokenizer
            _model = model
            _load_error = None
        except Exception as e:  # surfaced via /health
            _load_error = f"{type(e).__name__}: {e}"
            raise


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
_MOCK_ANSWER = (
    "MOCK MODE — the container is running without the 20B model loaded. "
    "This mode exists so the Flask app and Docker image can be verified on "
    "hardware without a GPU. Deploy with MOCK_MODE=0 on a CUDA GPU to get "
    "real answers from the fine-tuned MedBot."
)


def ask_medbot(question: str, max_new_tokens: int | None = None) -> dict:
    """Answer a question. Returns {'answer': str, 'warning': str | None}."""
    question = question.strip()
    warning = check_emergency(question)

    if MOCK_MODE:
        return {"answer": _MOCK_ANSWER, "warning": warning}

    load_model()

    import torch

    prompt = make_prompt(question)
    inputs = _tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    # Decode ONLY the new tokens (slice off the prompt), as in the notebook.
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    answer = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"answer": answer, "warning": warning}
