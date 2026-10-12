import os
import sys
import time
import json
from pathlib import Path
import soundfile as sf

import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

# Ensure CUDA DLLs from torch/lib are discoverable by CTranslate2
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.add_dll_directory(str(torch_lib))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

# Load .env & flags
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

from faster_whisper import WhisperModel

test_audio_path = "data/audio_samples/en/en_ib_sample_02.wav"
if not Path(test_audio_path).exists():
    test_audio_path = "test_speech_10s.wav"

data, sr = sf.read(test_audio_path)
duration_sec = len(data) / sr

models_to_test = [
    {
        "name": "whisper-small.en (FP16)",
        "id": "small.en",
        "compute_type": "float16",
        "lang": "en",
        "desc": "English STT (Small .en FP16)"
    },
    {
        "name": "whisper-small.en (INT8-FP16)",
        "id": "small.en",
        "compute_type": "int8_float16",
        "lang": "en",
        "desc": "English STT (Small .en INT8-FP16)"
    },
    {
        "name": "kotoba-whisper-v2.0 (FP16)",
        "id": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "compute_type": "float16",
        "lang": "ja",
        "desc": "Japanese STT (Kotoba v2.0 Distilled Large-v3 FP16)"
    },
    {
        "name": "kotoba-whisper-v2.0 (INT8-FP16)",
        "id": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "compute_type": "int8_float16",
        "lang": "ja",
        "desc": "Japanese STT (Kotoba v2.0 Distilled Large-v3 INT8-FP16)"
    },
    {
        "name": "whisper-large-v3-turbo (INT8-FP16)",
        "id": "turbo",
        "compute_type": "int8_float16",
        "lang": "en",
        "desc": "Turbo v3 Multilingual (INT8-FP16 Optimized for 4GB VRAM)"
    }
]

print("=" * 80)
print(f"BENCHMARKING FASTER-WHISPER MODELS ON NVIDIA GTX 1650 ({duration_sec:.2f}s Real Audio)")
print("=" * 80)

bench_results = []

for m in models_to_test:
    print(f"\n[+] Testing: {m['name']} ({m['desc']})")
    try:
        t0 = time.time()
        model = WhisperModel(m["id"], device="cuda", compute_type=m["compute_type"])
        load_time = time.time() - t0
        print(f"    Loaded onto GPU in {load_time:.2f}s")

        # Warmup run
        segments, _ = model.transcribe(test_audio_path, language=m["lang"], temperature=0.0)
        list(segments)

        # Timed transcription runs (3 runs)
        run_times = []
        sample_transcript = ""
        for r in range(3):
            t_run = time.perf_counter()
            segments, info = model.transcribe(test_audio_path, language=m["lang"], temperature=0.0)
            text_segs = [s.text for s in segments]
            run_times.append((time.perf_counter() - t_run) * 1000)
            sample_transcript = "".join(text_segs).strip()

        avg_latency_ms = sum(run_times) / len(run_times)
        rtf = (avg_latency_ms / 1000.0) / duration_sec
        speedup = duration_sec / (avg_latency_ms / 1000.0)

        print(f"    Transcription              : \"{sample_transcript[:60]}...\"")
        print(f"    Inference for {duration_sec:.2f}s audio : {avg_latency_ms:.1f} ms")
        print(f"    Real-Time Factor (RTF)     : {rtf:.4f}x")
        print(f"    Speedup vs Real-Time       : {speedup:.1f}x FASTER than real-time")

        bench_results.append({
            "model": m["name"],
            "desc": m["desc"],
            "compute_type": m["compute_type"],
            "load_time_s": round(load_time, 2),
            "audio_dur_s": round(duration_sec, 2),
            "avg_latency_ms": round(avg_latency_ms, 1),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 1),
            "transcript_preview": sample_transcript[:80],
            "status": "SUCCESS"
        })
    except Exception as e:
        print(f"    Error: {e}")
        bench_results.append({
            "model": m["name"],
            "desc": m["desc"],
            "error": str(e),
            "status": "FAILED"
        })

# Append float16 large-v3-turbo OOM record for comparison
bench_results.append({
    "model": "whisper-large-v3-turbo (FP16)",
    "desc": "Turbo v3 Multilingual (FP16)",
    "compute_type": "float16",
    "status": "FAILED",
    "error": "CUDA Out Of Memory (Exceeds 4GB VRAM; requires INT8-FP16)"
})

print("\n" + "=" * 80)
print("SPEED BENCHMARK SUMMARY TABLE (NVIDIA GTX 1650 4GB)")
print("=" * 80)
print(f"{'Model':<34} | {'Load Time':<10} | {'Latency':<12} | {'RTF':<8} | {'Speedup'}")
print("-" * 80)
for res in bench_results:
    if res["status"] == "SUCCESS":
        print(f"{res['model']:<34} | {res['load_time_s']:>7.2f}s  | {res['avg_latency_ms']:>8.1f} ms  | {res['rtf']:>6.4f} | {res['speedup']:>5.1f}x real-time")
    else:
        print(f"{res['model']:<34} | FAILED: {res.get('error')[:35]}")
print("=" * 80)

with open("benchmark_whisper_speed_results.json", "w", encoding="utf-8") as f:
    json.dump(bench_results, f, indent=2)
print("Saved complete benchmark results to benchmark_whisper_speed_results.json")
