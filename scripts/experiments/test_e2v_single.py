import os
import sys
import socket

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

# Load HF token if present
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

import torch
from funasr import AutoModel

print("Loading emotion2vec_plus_large...")
try:
    # Check if huggingface or modelscope works
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="ms") # or hub="hf"
    print("Model loaded successfully!")
    wav_path = "data/audio_samples/en/en_ib_sample_01.wav"
    res = model.generate(wav_path, output_dir="./scratch", granularity="utterance", extract_embedding=False)
    print("Inference result:", res)
except Exception as e:
    print("Failed with hub='ms':", e)
    print("Trying hub='hf'...")
    try:
        model = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf")
        print("Model loaded successfully via HF!")
        wav_path = "data/audio_samples/en/en_ib_sample_01.wav"
        res = model.generate(wav_path, output_dir="./scratch", granularity="utterance", extract_embedding=False)
        print("Inference result:", res)
    except Exception as e2:
        print("Failed with hub='hf':", e2)
