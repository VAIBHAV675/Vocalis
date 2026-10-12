import os
import sys
import time
import json
import gc
from pathlib import Path
import numpy as np

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load ground truth & existing benchmark
with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)
all_samples = gt_data["english_samples"] + gt_data["japanese_samples"]

with open("benchmark_60_audio_results.json", "r", encoding="utf-8") as f:
    primed_results_data = json.load(f)
primed_records = {r["sample_id"]: r for r in primed_results_data["records"]}

# Canonical 8 emotions schema
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the caller statement and classify into one of the 8 canonical emotions "
            "based on the true mental and emotional state."
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
    }
}

EMOTION2VEC_MAP = {
    "angry": "frustration_irritation",
    "disgusted": "frustration_irritation",
    "fearful": "anxiety_confusion",
    "happy": "satisfaction_gratitude",
    "neutral": "neutral_inquiry",
    "sad": "disappointed_sad",
    "surprised": "anxiety_confusion",
    "other": "neutral_inquiry"
}

import torch
from laya import Router

print("=" * 80)
print("FUSION COMPARISON EXPERIMENT ACROSS 60 REAL AUDIO RECORDINGS")
print("COMPARING: (1) Text-Only Laya vs (2) Natural Language Priming vs (3) Weighted Score Fusion")
print(f"DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 80)

router = Router(preload=False)
router.preload(["multilingual"])
print("Laya mmBERT router preloaded.\n")

results = []

for idx, s in enumerate(all_samples, 1):
    sid = s["sample_id"]
    lang = s["language"]
    expected = s["ground_truth_emotion"]
    transcript = s["ground_truth_transcript"]
    
    # 1. Text-Only Inference (Pure transcript, NO acoustic hints)
    text_prompt = f"Speaker transcript: 「{transcript}」"
    with torch.inference_mode():
        res_text = router.predict(text_prompt, IB_QUESTIONS, model="multilingual")
    
    text_pred = res_text["answers"]["detected_emotion"]["choice"]
    text_probs = res_text["answers"]["detected_emotion"]["probabilities"]
    text_conf = res_text["answers"]["detected_emotion"]["answer_confidence"]
    
    # 2. Natural Language Primed Result (from earlier benchmark)
    primed_rec = primed_records[sid]
    primed_pred = primed_rec["predicted_emotion"]
    primed_conf = primed_rec["laya_confidence"]
    ser_label = primed_rec["ser_prior"]
    ser_score = primed_rec["ser_confidence"]
    
    # 3. Weighted Score Fusion (e.g. 0.6 * Text Probabilities + 0.4 * Audio Probabilities)
    # Construct Audio Probability distribution over the 8 classes
    mapped_ser_class = EMOTION2VEC_MAP.get(ser_label.lower(), "neutral_inquiry")
    audio_probs = {c: 0.05 / 7 for c in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
    audio_probs[mapped_ser_class] = 0.95
    
    w_text = 0.6
    w_audio = 0.4
    fused_probs = {}
    for c in IB_QUESTIONS["detected_emotion"]["criteria"].keys():
        fused_probs[c] = (w_text * text_probs.get(c, 0.0)) + (w_audio * audio_probs.get(c, 0.0))
        
    fused_pred = max(fused_probs.items(), key=lambda x: x[1])[0]
    fused_conf = fused_probs[fused_pred]
    
    match_text = (text_pred == expected)
    match_primed = (primed_pred == expected)
    match_fused = (fused_pred == expected)
    
    results.append({
        "sample_id": sid,
        "lang": lang,
        "transcript": transcript,
        "expected": expected,
        "text_pred": text_pred,
        "text_match": match_text,
        "primed_pred": primed_pred,
        "primed_match": match_primed,
        "fused_pred": fused_pred,
        "fused_match": match_fused,
        "ser_label": ser_label,
        "ser_score": ser_score
    })
    
    m_t = "PASS" if match_text else "FAIL"
    m_p = "PASS" if match_primed else "FAIL"
    m_f = "PASS" if match_fused else "FAIL"
    print(f"[{idx:02d}/60] {sid} | Exp: {expected:<22} | Text: {text_pred:<22} [{m_t}] | Primed: {primed_pred:<22} [{m_p}] | Fused: {fused_pred:<22} [{m_f}]")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Summary Calculations
total = len(results)
en_res = [r for r in results if r["lang"] == "en"]
ja_res = [r for r in results if r["lang"] == "ja"]

acc_text_total = sum(1 for r in results if r["text_match"]) / total
acc_primed_total = sum(1 for r in results if r["primed_match"]) / total
acc_fused_total = sum(1 for r in results if r["fused_match"]) / total

acc_text_en = sum(1 for r in en_res if r["text_match"]) / len(en_res)
acc_primed_en = sum(1 for r in en_res if r["primed_match"]) / len(en_res)
acc_fused_en = sum(1 for r in en_res if r["fused_match"]) / len(en_res)

acc_text_ja = sum(1 for r in ja_res if r["text_match"]) / len(ja_res)
acc_primed_ja = sum(1 for r in ja_res if r["primed_match"]) / len(ja_res)
acc_fused_ja = sum(1 for r in ja_res if r["fused_match"]) / len(ja_res)

print("\n" + "=" * 80)
print("FINAL FUSION METHOD COMPARISON SUMMARY (60 AUDIO RECORDINGS)")
print("=" * 80)
print(f"{'Method':<28} | {'Overall (60)':<14} | {'English (30)':<14} | {'Japanese (30)':<14}")
print("-" * 76)
print(f"{'1. Text-Only Laya':<28} | {acc_text_total:>5.1%} ({sum(1 for r in results if r['text_match'])}/60)    | {acc_text_en:>5.1%} ({sum(1 for r in en_res if r['text_match'])}/30)    | {acc_text_ja:>5.1%} ({sum(1 for r in ja_res if r['text_match'])}/30)")
print(f"{'2. Natural Language Priming':<28} | {acc_primed_total:>5.1%} ({sum(1 for r in results if r['primed_match'])}/60)    | {acc_primed_en:>5.1%} ({sum(1 for r in en_res if r['primed_match'])}/30)    | {acc_primed_ja:>5.1%} ({sum(1 for r in ja_res if r['primed_match'])}/30)")
print(f"{'3. Weighted Score (0.6/0.4)':<28} | {acc_fused_total:>5.1%} ({sum(1 for r in results if r['fused_match'])}/60)    | {acc_fused_en:>5.1%} ({sum(1 for r in en_res if r['fused_match'])}/30)    | {acc_fused_ja:>5.1%} ({sum(1 for r in ja_res if r['fused_match'])}/30)")
print("=" * 80)

# Save to json
with open("fusion_comparison_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "overall": {
            "text_only_acc": acc_text_total,
            "primed_acc": acc_primed_total,
            "weighted_fused_acc": acc_fused_total
        },
        "english": {
            "text_only_acc": acc_text_en,
            "primed_acc": acc_primed_en,
            "weighted_fused_acc": acc_fused_en
        },
        "japanese": {
            "text_only_acc": acc_text_ja,
            "primed_acc": acc_primed_ja,
            "weighted_fused_acc": acc_fused_ja
        },
        "records": results
    }, f, indent=2, ensure_ascii=False)
print("Saved complete comparison to fusion_comparison_results.json")
