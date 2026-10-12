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

# Load ground truth & existing benchmark records
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
            "Analyze the caller's spoken statement in the full context of their vocal acoustics and delivery. "
            "Resolve any tension between the words and vocal tone (e.g. sarcasm, polite complaints, subdued relief). "
            "Classify the customer's true emotional state into one of the 8 canonical categories."
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

def describe_prosody_naturally(raw: dict) -> dict:
    mean_f0 = raw.get("mean_f0", 120.0)
    std_f0 = raw.get("std_f0", 20.0)
    rms_db = raw.get("rms_db", 65.0)
    syll_rate = raw.get("syll_rate", 4.5)
    centroid = raw.get("centroid", 1500.0)
    flatness = raw.get("flatness", 0.01)

    # Pitch Level & Contour Significance
    if mean_f0 == 0:
        pitch_desc = "unvoiced or barely audible whispering"
    elif mean_f0 > 240:
        pitch_desc = "exceptionally high and sharp, reflecting intense emotional arousal"
    elif mean_f0 > 175:
        pitch_desc = "elevated in pitch, suggesting agitation, alertness, or heightened tension"
    elif mean_f0 < 100:
        pitch_desc = "notably low and depressed in frequency, conveying subdued or dejected gravity"
    else:
        pitch_desc = "steady and at a normal conversational frequency"

    if std_f0 < 15:
        contour_desc = "delivered with a flat, monotonic contour showing minimal inflection"
    elif std_f0 > 45:
        contour_desc = "marked by volatile pitch fluctuations and dynamic vocal leaps"
    else:
        contour_desc = "natural and smooth melodic modulation"

    # Speaking Tempo / Cadence Significance
    if syll_rate > 6.0:
        tempo_desc = "spoken rapidly at an urgent, hurried cadence"
    elif syll_rate < 3.2:
        tempo_desc = "spoken slowly and deliberately with measured pauses"
    else:
        tempo_desc = "delivered at a comfortable, standard business conversation speed"

    # Volume / Acoustic Energy Significance
    if rms_db > 78:
        energy_desc = "loud and forceful, carrying high vocal intensity"
    elif rms_db < 58:
        energy_desc = "quiet and subdued, with low vocal projection"
    else:
        energy_desc = "clear and balanced at moderate conversational volume"

    # Voice Quality / Phonation Significance
    if flatness > 0.04:
        vq_desc = "breathy and soft vocal phonation"
    elif centroid > 2400:
        vq_desc = "tense, bright, and strained vocal resonance"
    elif centroid < 1100:
        vq_desc = "relaxed, deep chest resonance"
    else:
        vq_desc = "clear modal phonation with stable harmonic resonance"

    return {
        "pitch_desc": f"{pitch_desc}, {contour_desc}",
        "tempo_desc": tempo_desc,
        "energy_desc": energy_desc,
        "vq_desc": vq_desc
    }

def format_coherent_multimodal_prompt(transcript: str, raw_prosody: dict, ser_prior: str, ser_conf: float) -> str:
    p = describe_prosody_naturally(raw_prosody)
    
    ser_map = {
        "neutral": "composed, neutral, and matter-of-fact",
        "happy": "positive, upbeat, and satisfied",
        "angry": "hostile, irritated, or abrasive",
        "sad": "sorrowful, dejected, or disappointed",
        "fearful": "anxious, fearful, or nervous",
        "surprised": "startled or astonished",
        "disgusted": "displeased or dismissive"
    }
    ser_desc = ser_map.get(ser_prior.lower(), f"{ser_prior}")

    return (
        f"A customer made the following statement during a business discussion:\n"
        f"「{transcript}」\n\n"
        f"Vocal acoustic analysis of the recording indicates:\n"
        f"The pitch is {p['pitch_desc']}. "
        f"The delivery speed is {p['tempo_desc']}, with vocal volume that is {p['energy_desc']}. "
        f"The voice quality shows {p['vq_desc']}. "
        f"Speech acoustic models evaluate the caller's general vocal tone as {ser_desc} (confidence: {ser_conf:.0%}).\n\n"
        f"Synthesizing both the literal spoken words and the nuanced acoustic indicators, classify the caller's genuine emotional state."
    )

import torch
from laya import Router

print("=" * 80)
print("BENCHMARK: COHERENT NATURAL LANGUAGE MULTIMODAL PROMPT")
print("DESIGN: Fluent Semantic Narrative with Natural Acoustic Significance Placeholders")
print(f"DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 80)

router = Router(preload=False)
router.preload(["multilingual"])
print("Laya mmBERT router preloaded.\n")

print(f"{'ID':<16} | {'Lang':<4} | {'Expected':<22} | {'Predicted':<22} | {'Conf':<6} | {'Match'}")
print("-" * 88)

results = []
latencies = []

for idx, s in enumerate(all_samples, 1):
    sid = s["sample_id"]
    lang = s["language"]
    expected = s["ground_truth_emotion"]
    transcript = s["ground_truth_transcript"]
    rec = existing_records[sid]
    raw_prosody = rec["prosody"]["raw_features"]
    ser_label = rec["ser_prior"]
    ser_score = rec["ser_confidence"]
    
    prompt = format_coherent_multimodal_prompt(transcript, raw_prosody, ser_label, ser_score)
    
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
        "prompt_used": prompt
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
print("FINAL RESULTS: COHERENT NATURAL LANGUAGE MULTIMODAL PROMPT (60 AUDIO CALLS)")
print("=" * 80)
print(f"Overall Accuracy Across 60 Calls : {acc_total:.1%} ({sum(1 for r in results if r['match'])}/60)")
print(f"English Accuracy (30 Calls)      : {acc_en:.1%} ({sum(1 for r in en_res if r['match'])}/30)")
print(f"Japanese Accuracy (30 Calls)     : {acc_ja:.1%} ({sum(1 for r in ja_res if r['match'])}/30)")
print(f"Average Inference Latency (CUDA) : {avg_lat:.1f} ms")
print("=" * 80)

with open("benchmark_coherent_natural_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "overall_accuracy": acc_total,
        "english_accuracy": acc_en,
        "japanese_accuracy": acc_ja,
        "average_latency_ms": avg_lat,
        "records": results
    }, f, indent=2, ensure_ascii=False)
print("Saved complete benchmark results to benchmark_coherent_natural_results.json")
