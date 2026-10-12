import os
import sys
sys.path.insert(0, './kev')

import socket
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

# Load HF token
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from kev.checkpoint import Checkpoint, LoadOptions

print("Testing Checkpoint resolution for jaredpalmer/kev-0.8b...")
ck = Checkpoint("jaredpalmer/kev-0.8b")
print(f"Base model: {ck.meta.base}")
print(f"Temperature: {ck.meta.temperature}")
print(f"Weights dtype: {ck.meta.weights_dtype}")
print(f"Lora rank: {ck.meta.lora}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading tokenizer and model onto {device}...")
opts = LoadOptions(dtype=torch.float16, merge=True)
tok, model = ck.load(device, opts)
print("Kev-0.8B successfully loaded into memory!")
