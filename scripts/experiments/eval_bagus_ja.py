import os
import sys
import json
import time
import torch
from pathlib import Path
from transformers import pipeline

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, flags=0: _orig_getaddrinfo(h, p, socket.AF_INET, t, pr, flags)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading Bagus Japanese SER model on {device}...")

pipe_ja = pipeline(
    "audio-classification",
    model="Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition",
    device=device
)
print("Bagus Japanese SER loaded successfully!")
print("Model classes:", pipe_ja.model.config.id2label)

# Load ground truth 30 Japanese samples
with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
ja_samples = gt_data["japanese_samples"]

# Emotion mapping from Bagus 4-class taxonomy to Canonical 8-class taxonomy
# ang -> outrage_hostility / frustration_irritation
# joy -> satisfaction_gratitude
# neu -> neutral_inquiry
# sad -> disappointed_sad
bagus_map = {
    "ang": ["frustration_irritation", "outrage_hostility"],
    "joy": ["satisfaction_gratitude"],
    "neu": ["neutral_inquiry"],
    "sad": ["disappointed_sad"]
}

print("\n--- EVALUATING BAGUS ON 30 JAPANESE RECORDINGS ---")
print(f"{'ID':<16} | {'Expected':<24} | {'Bagus Raw':<10} | {'Score':<6} | {'Match':<6} | {'Lat(ms)':<7}")
print("-" * 80)

correct = 0
latencies = []
results = []

for s in ja_samples:
    sid = s["sample_id"]
    wav_path = s["wav_path"]
    expected = s["ground_truth_emotion"]
    
    t0 = time.perf_counter()
    res = pipe_ja(wav_path, top_k=1)[0]
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    
    raw_label = res["label"]
    score = res["score"]
    
    matched_canonicals = bagus_map.get(raw_label, [])
    is_match = (expected in matched_canonicals)
    if is_match:
        correct += 1
    
    match_str = "[PASS]" if is_match else "[FAIL]"
    print(f"{sid:<16} | {expected:<24} | {raw_label:<10} | {score:>5.1%} | {match_str:<6} | {lat:>6.1f}ms")
    
    results.append({
        "sample_id": sid,
        "expected": expected,
        "bagus_raw": raw_label,
        "score": score,
        "match": is_match,
        "latency_ms": lat
    })

avg_lat = sum(latencies) / len(latencies)
acc = correct / len(ja_samples)

print("=" * 80)
print(f"Bagus Japanese SER Accuracy: {correct}/{len(ja_samples)} ({acc:.1%})")
print(f"Average Latency: {avg_lat:.1f}ms")
print("=" * 80)

# Save result
with open("bagus_japanese_ser_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": "Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition",
        "accuracy": acc,
        "correct": correct,
        "total": len(ja_samples),
        "avg_latency_ms": avg_lat,
        "results": results
    }, f, indent=2)
