import os
import sys
import json
import time
import torch
import librosa
from pathlib import Path
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, flags=0: _orig_getaddrinfo(h, p, socket.AF_INET, t, pr, flags)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading ehcalabres English SER model on {device}...")

model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feat = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
model.eval()

print("English SER loaded successfully!")
id2label = model.config.id2label
print("Model classes:", id2label)

# Load ground truth 30 English samples
with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
en_samples = gt_data["english_samples"]

# Mapping from RAVDESS to Canonical 8-class taxonomy
# calm, neutral -> neutral_inquiry
# happy -> satisfaction_gratitude
# angry -> frustration_irritation / outrage_hostility
# sad -> disappointed_sad
# fearful, surprised -> anxiety_confusion / urgency_panic
ehcalabres_map = {
    "calm": ["neutral_inquiry"],
    "neutral": ["neutral_inquiry"],
    "happy": ["satisfaction_gratitude"],
    "angry": ["frustration_irritation", "outrage_hostility"],
    "sad": ["disappointed_sad"],
    "fearful": ["anxiety_confusion", "urgency_panic"],
    "surprised": ["anxiety_confusion", "urgency_panic"],
    "disgust": ["frustration_irritation"]
}

print("\n--- EVALUATING EHCALABRES ON 30 ENGLISH EARNINGS22 CALLS ---")
print(f"{'ID':<16} | {'Expected':<24} | {'Raw SER':<12} | {'Score':<6} | {'Match':<6} | {'Lat(ms)':<7}")
print("-" * 84)

correct = 0
latencies = []
results = []

for s in en_samples:
    sid = s["sample_id"]
    wav_path = f"data/audio_samples/en/{s['filename']}"
    expected = s["ground_truth_emotion"]
    
    y, sr = librosa.load(wav_path, sr=16000)
    inputs = feat(y, sampling_rate=16000, return_tensors="pt").to(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    
    best_id = int(torch.argmax(probs))
    raw_label = id2label[best_id]
    score = float(probs[best_id])
    
    matched_canonicals = ehcalabres_map.get(raw_label, [])
    is_match = (expected in matched_canonicals)
    if is_match:
        correct += 1
    
    match_str = "[PASS]" if is_match else "[FAIL]"
    print(f"{sid:<16} | {expected:<24} | {raw_label:<12} | {score:>5.1%} | {match_str:<6} | {lat:>6.1f}ms")
    
    results.append({
        "sample_id": sid,
        "expected": expected,
        "ehcalabres_raw": raw_label,
        "score": score,
        "match": is_match,
        "latency_ms": lat
    })

avg_lat = sum(latencies) / len(latencies)
acc = correct / len(en_samples)

print("=" * 84)
print(f"ehcalabres English SER Standalone Accuracy: {correct}/{len(en_samples)} ({acc:.1%})")
print(f"Average Latency: {avg_lat:.1f}ms")
print("=" * 84)

# Save result
with open("ehcalabres_english_ser_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "accuracy": acc,
        "correct": correct,
        "total": len(en_samples),
        "avg_latency_ms": avg_lat,
        "results": results
    }, f, indent=2)
