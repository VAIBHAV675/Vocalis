import os
import sys
import socket
from pathlib import Path

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

# Load .env
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

print(f"HF_TOKEN available: {'HF_TOKEN' in os.environ}")

from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test Japanese Bagus wav2vec2-xlsr model (360 MB)
print("\n--- Testing Japanese SER on Hugging Face (Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition) ---")
try:
    pipe_ja = pipeline("audio-classification", model="Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition", device=device)
    res_ja = pipe_ja("data/audio_samples/ja/ja_ib_sample_01.wav")
    print("Japanese SER Output on ja_ib_sample_01:", res_ja)
except Exception as e:
    print("Japanese model error:", e)

# Test English harshit345 xlsr model (360 MB) or ehcalabres
print("\n--- Testing English SER on Hugging Face (harshit345/xlsr-wav2vec-speech-emotion-recognition) ---")
try:
    pipe_en = pipeline("audio-classification", model="harshit345/xlsr-wav2vec-speech-emotion-recognition", device=device)
    res_en = pipe_en("data/audio_samples/en/en_ib_sample_01.wav")
    print("English SER Output on en_ib_sample_01:", res_en)
except Exception as e:
    print("English model error:", e)
