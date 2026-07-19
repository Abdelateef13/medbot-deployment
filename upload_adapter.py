"""
upload_adapter.py — run this ONCE (in Colab, next to your Drive mount)
to publish the trained LoRA adapter + tokenizer to the Hugging Face Hub.

The deployed app then downloads the adapter from the Hub at startup
(ADAPTER_REPO env var), so the ~30MB adapter never needs to live inside
the Docker image and Google Drive is not needed in production.

Usage in Colab:
    !pip -q install huggingface_hub
    from huggingface_hub import notebook_login
    notebook_login()          # paste a WRITE token from hf.co/settings/tokens
    %run upload_adapter.py
"""

from huggingface_hub import HfApi

# 1. Where the notebook saved the adapter (cell #19)
LOCAL_DIR = "/content/drive/MyDrive/medbot_lora"

# 2. Your Hub repo — CHANGE the username
REPO_ID = "YOUR_HF_USERNAME/medbot-lora"

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

# Upload only what inference needs: adapter weights/config + tokenizer.
# Checkpoints (checkpoint-*/) are excluded — they are training state.
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=["checkpoint-*", "runs/*", "*.bin.lock"],
)

print(f"Done → https://huggingface.co/{REPO_ID}")
