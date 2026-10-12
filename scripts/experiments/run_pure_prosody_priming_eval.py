import os
import sys
import time
import json
import gc
from pathlib import Path

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load ground truth & existing prosody features
with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
all_samples = gt_data["english_samples"] + gt_data["japanese_samples"]

with open("benchmark_60_audio_results.json", "r", encoding="utf-8") as f:
    existing_data = json.load(f)
existing_records = {r["sample_id"]: r for r in existing_data["records"]}

# Canonical 8 emotions schema
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the speaker transcript alongside acoustic prosodic cues (pitch, tempo, energy, voice quality). "
            "Resolve any nuance between the literal words and vocal delivery. "
            "Classify into one of the 8 canonical emotions based on the speaker's true mental and emotional state."
        ),
        "criteria": {
            "satisfaction_gratitude": "Delight, genuine satisfaction, heartfelt gratitude, relief, happiness, warm appreciation.",
            "neutral_inquiry": "Calm routine business inquiry, objective questions, matter-of-fact tone, operational transaction queries.",
            "frustration_irritation": "Annoyance, impatient complaints, repeated friction, irritation, vocal exasperation.",
            "outrage_hostility": "Extreme fury, aggressive hostility, shouting, legal threats, intense confrontation.",
            "disappointed_sad": "Sorrow, deep sadness, grief, emotional dejection, heavy sighing, depressed hopeless tone.",
            "sarcasm_passive_aggression": "Ironic praise, mocking cynicism, polite words concealing sharp anger and contempt.",
            "anxiety_confusion": "Nervous panic, trembling uncertainty, fear, worried distress, helpless confusion.",
            "urgency_panic": "Critical crisis, urgent emergency, immediate stop or freeze demand, frantic alarm."
        }
    },
    "sarcasm_detected": {
        "type": "noul",
        "instructions": "Is the speaker using polite or formal business language sarcastically to mask severe hostility?"
    }
}

def format_pure_prosodic_prompt(transcript: str, prosody: dict) -> str:
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic prosodic cues:\n"
        f"- Pitch contour: {prosody['pitch_str']}\n"
        f"- Speaking tempo: {prosody['tempo_str']}\n"
        f"- Energy & vocal intensity: {prosody['energy_str']}\n"
        f"- Voice quality: {prosody['voice_quality_str']}"
    )

import torch
from laya import Router

print("=" * 80)
print("BENCHMARK: PURE ACOUSTIC-PROSODIC NATURAL LANGUAGE PRIMING (WITHOUT COARSE SER PRIOR)")
print("PROMPT STRUCTURE: Transcript + Librosa Physical Acoustics (Pitch, Tempo, Energy, Quality)")
print(f"DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 80)

router = Router(preload=False)
router.preload(["multilingual"])
print("Laya mmBERT preloaded.\n")

print(f"{'ID':<16} | {'Lang':<4} | {'Expected':<22} | {'Predicted':<22} | {'Conf':<6} | {'Match'}")
print("-" * 88)

results = []
latencies = []

for idx, s in enumerate(all_samples, 1):
    sid = s["sample_id"]
    lang = s["language"]
    expected = s["ground_truth_emotion"]
    transcript = s["ground_truth_transcript"]
    prosody_info = existing_records[sid]["prosody"]
    
    prompt = format_pure_prosodic_prompt(transcript, prosody_info)
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        res = router.predict(prompt, IB_QUESTIONS, model="multilingual")
    lat_ms = (time.perf_counter() - t0) * 1000
    latencies.append(lat_ms)
    
    pred_emo = res["answers"]["detected_emotion"]["choice"]
    pred_conf = res["answers"]["detected_emotion"]["answer_confidence"]
    sarc_p = res["answers"]["sarcasm_detected"]["noul"]
    match = (pred_emo == expected)
    
    m_str = "[PASS]" if match else "[FAIL]"
    print(f"{sid:<16} | {lang:<4} | {expected:<22} | {pred_emo:<22} | {pred_conf:>5.1%} | {m_str}")
    
    results.append({
        "sample_id": sid,
        "language": lang,
        "transcript": transcript,
        "expected_emotion": expected,
        "predicted_emotion": pred_emo,
        "confidence": pred_conf,
        "sarcasm_p": sarc_p,
        "match": match,
        "latency_ms": lat_ms,
        "prosody": prosody_info
    })
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Metrics
en_res = [r for r in results if r["language"] == "en"]
ja_res = [r for r in results if r["language"] == "ja"]

acc_total = sum(1 for r in results if r["match"]) / len(results)
acc_en = sum(1 for r in en_res if r["match"]) / len(en_res)
acc_ja = sum(1 for r in ja_res if r["match"]) / len(ja_res)
avg_lat = sum(latencies) / len(latencies)

print("\n" + "=" * 80)
print("FINAL RESULTS: PURE PROSODIC PRIMING (WITHOUT COARSE SER PRIOR)")
print("=" * 80)
print(f"Overall Accuracy Across 60 Calls : {acc_total:.1%} ({sum(1 for r in results if r['match'])}/60)")
print(f"English Accuracy (30 Calls)      : {acc_en:.1%} ({sum(1 for r in en_res if r['match'])}/30)")
print(f"Japanese Accuracy (30 Calls)     : {acc_ja:.1%} ({sum(1 for r in ja_res if r['match'])}/30)")
print(f"Average Inference Latency (CUDA) : {avg_lat:.1f} ms")
print("=" * 80)

with open("benchmark_pure_prosody_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "overall_accuracy": acc_total,
        "english_accuracy": acc_en,
        "japanese_accuracy": acc_ja,
        "average_latency_ms": avg_lat,
        "records": results
    }, f, indent=2, ensure_ascii=False)
print("Saved complete benchmark results to benchmark_pure_prosody_results.json")
