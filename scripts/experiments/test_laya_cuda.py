import os
import sys
import time
from pathlib import Path

import socket
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

import torch
print("=" * 60)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

from laya import Router

router = Router(preload=False)
t0 = time.time()
router.preload(["multilingual"])
print(f"Laya preload finished in {time.time() - t0:.2f}s")

# Test forward pass latency
t_start = time.perf_counter()
res = router.predict(
    "The syndicate syndication was finalized and closed at par. Excellent execution by your M&A team.",
    {
        "emotion": {
            "type": "choice",
            "instructions": "Classify call emotion",
            "criteria": {
                "satisfaction": "Successful deal, relief, gratitude",
                "frustration": "Issues and friction"
            }
        }
    },
    model="multilingual"
)
latency_ms = (time.perf_counter() - t_start) * 1000
print(f"Laya forward pass latency: {latency_ms:.1f} ms")
print(f"Result: {res}")
