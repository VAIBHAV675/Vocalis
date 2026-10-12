import os
import sys
import time
import json
from pathlib import Path
import soundfile as sf

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

# CUDA DLL discovery
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.add_dll_directory(str(torch_lib))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

en_audio = "data/audio_samples/en/en_ib_sample_02.wav"
ja_audio = "data/audio_samples/ja/ja_ib_sample_01.wav"

data_en, sr_en = sf.read(en_audio)
dur_en = len(data_en) / sr_en

data_ja, sr_ja = sf.read(ja_audio)
dur_ja = len(data_ja) / sr_ja

print("=" * 80)
print(f"INT8 CUDA SPEED BENCHMARK (NVIDIA GTX 1650 4GB)")
print(f"EN Test Audio: {en_audio} ({dur_en:.2f}s)")
print(f"JA Test Audio: {ja_audio} ({dur_ja:.2f}s)")
print("=" * 80)

models_config = [
    {
        "name": "whisper-small.en",
        "id": "small.en",
        "audio": en_audio,
        "dur": dur_en,
        "lang": "en",
        "kwargs": {},
        "desc": "English STT (Small .en INT8)"
    },
    {
        "name": "kotoba-whisper-v2.0",
        "id": "kotoba-tech/kotoba-whisper-v2.0-faster",
        "audio": ja_audio,
        "dur": dur_ja,
        "lang": "ja",
        "kwargs": {"chunk_length": 15, "condition_on_previous_text": False},
        "desc": "Japanese STT (Kotoba v2.0 Distilled Large-v3 INT8)"
    },
    {
        "name": "whisper-large-v3-turbo (EN)",
        "id": "turbo",
        "audio": en_audio,
        "dur": dur_en,
        "lang": "en",
        "kwargs": {},
        "desc": "Multilingual Turbo (EN Audio INT8)"
    },
    {
        "name": "whisper-large-v3-turbo (JA)",
        "id": "turbo",
        "audio": ja_audio,
        "dur": dur_ja,
        "lang": "ja",
        "kwargs": {},
        "desc": "Multilingual Turbo (JA Audio INT8)"
    }
]

benchmark_records = []

for cfg in models_config:
    print(f"\n[+] Testing: {cfg['name']} ({cfg['desc']})")
    t0 = time.time()
    model = WhisperModel(cfg["id"], device="cuda", compute_type="int8_float16")
    load_time = time.time() - t0
    print(f"    Loaded onto GPU in {load_time:.2f}s")

    # Warmup run
    segments, _ = model.transcribe(cfg["audio"], language=cfg["lang"], temperature=0.0, **cfg["kwargs"])
    list(segments)

    # 3 Timed runs
    latencies = []
    text_result = ""
    for r in range(3):
        t_start = time.perf_counter()
        segments, info = model.transcribe(cfg["audio"], language=cfg["lang"], temperature=0.0, **cfg["kwargs"])
        text_parts = [s.text for s in segments]
        latencies.append((time.perf_counter() - t_start) * 1000)
        text_result = "".join(text_parts).strip()

    avg_lat = sum(latencies) / len(latencies)
    rtf = (avg_lat / 1000.0) / cfg["dur"]
    speedup = cfg["dur"] / (avg_lat / 1000.0)

    print(f"    Transcription              : \"{text_result[:65]}...\"")
    print(f"    Inference for {cfg['dur']:.2f}s audio : {avg_lat:.1f} ms")
    print(f"    Real-Time Factor (RTF)     : {rtf:.4f}x")
    print(f"    Speedup vs Real-Time       : {speedup:.1f}x FASTER than real-time")

    benchmark_records.append({
        "model": cfg["name"],
        "desc": cfg["desc"],
        "language": cfg["lang"],
        "audio_dur_s": round(cfg["dur"], 2),
        "load_time_s": round(load_time, 2),
        "avg_latency_ms": round(avg_lat, 1),
        "rtf": round(rtf, 4),
        "speedup": round(speedup, 1),
        "transcript": text_result
    })

print("\n" + "=" * 80)
print("INT8 SPEED BENCHMARK SUMMARY (NVIDIA GTX 1650 4GB)")
print("=" * 80)
print(f"{'Model':<30} | {'Lang':<4} | {'Audio Dur':<9} | {'Latency':<11} | {'RTF':<8} | {'Speedup'}")
print("-" * 80)
for rec in benchmark_records:
    print(f"{rec['model']:<30} | {rec['language']:<4} | {rec['audio_dur_s']:>6.2f}s   | {rec['avg_latency_ms']:>8.1f} ms | {rec['rtf']:>6.4f} | {rec['speedup']:>5.1f}x real-time")
print("=" * 80)

with open("benchmark_int8_results.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_records, f, indent=2, ensure_ascii=False)
print("Saved benchmark results to benchmark_int8_results.json")
