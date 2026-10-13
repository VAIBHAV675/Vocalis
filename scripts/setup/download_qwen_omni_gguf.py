import os
import socket
import time
from pathlib import Path

# Force IPv4 for fast HuggingFace CDN transfer
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
env_file = ROOT / ".env"
token = None
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = token

from huggingface_hub import hf_hub_download, HfApi

def main():
    models_dir = ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "ggml-org/Qwen2.5-Omni-3B-GGUF"
    files = [
        "Qwen2.5-Omni-3B-Q4_K_M.gguf",
        "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf",
    ]

    api = HfApi()
    infos = list(api.get_paths_info(repo_id, files))
    for info in infos:
        print(f"[Info] {repo_id}/{info.path}: {info.size / 1e9:.2f} GB")

    for fname in files:
        t0 = time.time()
        print(f"\n[+] Downloading {fname} from {repo_id} to {models_dir}...")
        out_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=str(models_dir),
            token=token,
        )
        print(f"[✓] Downloaded {fname} -> {out_path} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
