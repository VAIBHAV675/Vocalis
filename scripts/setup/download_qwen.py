import os
from pathlib import Path
import socket

# Force IPv4
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

env_file = Path(__file__).parent / ".env"
token = None
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = token

print(f"HF Token loaded: {'Yes' if token else 'No'}")

from huggingface_hub import hf_hub_download

print("Downloading Qwen3.5-0.8B-Base model files...")
try:
    path = hf_hub_download(
        repo_id="Qwen/Qwen3.5-0.8B-Base",
        filename="model.safetensors-00001-of-00001.safetensors",
        token=token,
    )
    print(f"Downloaded shard to: {path}")
except Exception as e:
    print(f"Error downloading: {e}")
