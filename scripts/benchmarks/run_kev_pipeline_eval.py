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

# Add local kev clone to path
sys.path.insert(0, str(Path(__file__).parent / "kev"))

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from kev.api import SystemOneRequest, to_record, to_answers
from kev.model import admit

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

def format_state(transcript: str, ser_prior: str, ser_conf: float, prosody: dict) -> str:
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic speech emotion prior: {ser_prior} (confidence: {ser_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {prosody.get('pitch_str', 'Normal')}\n"
        f"- Speech tempo: {prosody.get('tempo_str', 'Normal')}\n"
        f"- Energy & vocal intensity: {prosody.get('energy_str', 'Normal')}\n"
        f"- Voice quality: {prosody.get('voice_quality_str', 'Normal')}"
    )

def main():
    print("=" * 80)
    print("KEV-0.8B DECISION MODEL EVALUATION (SPEECH EMOTION RECOGNITION PIPELINE)")
    print("Target: jaredpalmer/kev-0.8b (Qwen3.5-0.8B + LoRA + Pointer Readout)")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load dataset & baseline results
    with open("benchmark_60_audio_results.json", "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    samples = baseline_data["records"]

    # Load ground truth transcripts
    with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    transcripts = {s["sample_id"]: s["ground_truth_transcript"] for s in (gt_data["english_samples"] + gt_data["japanese_samples"])}

    print(f"[+] Loaded {len(samples)} benchmark records (30 EN, 30 JA) with transcripts & extracted acoustics & SER priors")

    # 2. Load Kev-0.8B
    print("\n--- LOADING KEV-0.8B CHECKPOINT ---")
    t0 = time.time()
    ck = Checkpoint("jaredpalmer/kev-0.8b")
    print(f"[+] Resolved checkpoint: base={ck.meta.base}, lora={ck.meta.lora}, temp={ck.meta.temperature:.2f}")
    
    # Load on GPU with float16 to preserve VRAM
    opts = LoadOptions(dtype=torch.float16, merge=True, attn="sdpa")
    tok, model = ck.load(device, opts)
    print(f"[+] Kev-0.8B successfully loaded into {device} in {time.time() - t0:.2f}s")

    # 3. Inference evaluation loop
    print("\n--- RUNNING KEV-0.8B DECISION INFERENCE ACROSS ALL SAMPLES ---")
    print(f"{'ID':<16} | {'Lang':<4} | {'Expected':<22} | {'Kev Prediction':<22} | {'Conf':<6} | {'Match':<6} | {'Lat(ms)':<7}")
    print("-" * 96)

    records = []
    latencies = []
    contradictions = 0

    for idx, s in enumerate(samples, 1):
        sid = s["sample_id"]
        lang = s["language"]
        expected = s["expected_emotion"]
        transcript = transcripts.get(sid, "")
        ser_prior = s["ser_prior"]
        ser_conf = s["ser_confidence"]
        prosody = s["prosody"]

        state_text = format_state(transcript, ser_prior, ser_conf, prosody)
        req = SystemOneRequest(state=state_text, model="kev-latest", questions=IB_QUESTIONS)

        rec, meta = to_record(req)
        enc = admit(model, tok, rec)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            ps = model.probs(enc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)

        answers = to_answers([p.tolist() for p in ps], meta)
        pred_emo = answers["detected_emotion"]["choice"]
        pred_conf = answers["detected_emotion"]["confidence"]
        sarc_p = answers["sarcasm_detected"]["noul"]

        match = (pred_emo == expected)
        match_str = "[PASS]" if match else "[FAIL]"

        # Check if Kev contradicted SER prior
        contradicted_ser = (pred_emo != ser_prior)
        if contradicted_ser:
            contradictions += 1

        print(f"{sid:<16} | {lang:<4} | {expected:<22} | {pred_emo:<22} | {pred_conf:>5.1%} | {match_str:<6} | {lat:>6.1f}ms")

        records.append({
            "sample_id": sid,
            "language": lang,
            "expected_emotion": expected,
            "predicted_emotion": pred_emo,
            "kev_confidence": pred_conf,
            "sarcasm_probability": sarc_p,
            "match": match,
            "ser_prior": ser_prior,
            "ser_confidence": ser_conf,
            "contradicted_ser": contradicted_ser,
            "latency_ms": lat,
            "state_length_tokens": len(enc["ids"])
        })

    # 4. Summary metrics
    en_records = [r for r in records if r["language"] == "en"]
    ja_records = [r for r in records if r["language"] == "ja"]

    en_acc = sum(1 for r in en_records if r["match"]) / len(en_records) if en_records else 0.0
    ja_acc = sum(1 for r in ja_records if r["match"]) / len(ja_records) if ja_records else 0.0
    all_acc = sum(1 for r in records if r["match"]) / len(records)
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY: KEV-0.8B vs LAYA mmBERT")
    print("=" * 80)
    print(f"Overall Accuracy:  Kev-0.8B: {all_acc:.1%} ({sum(r['match'] for r in records)}/{len(records)}) | Laya Baseline: 43.3% (26/60)")
    print(f"English Accuracy:  Kev-0.8B: {en_acc:.1%} ({sum(r['match'] for r in en_records)}/{len(en_records)}) | Laya Baseline: 50.0% (15/30)")
    print(f"Japanese Accuracy: Kev-0.8B: {ja_acc:.1%} ({sum(r['match'] for r in ja_records)}/{len(ja_records)}) | Laya Baseline: 36.7% (11/30)")
    print(f"Average Latency:   Kev-0.8B: {avg_lat:.1f}ms | Laya Baseline: ~1200ms")
    print(f"SER Contradiction: Kev contradicted SER in {contradictions}/{len(records)} cases ({contradictions/len(records):.1%})")
    print("=" * 80)

    # Class breakdown
    classes = list(IB_QUESTIONS["detected_emotion"]["criteria"].keys())
    print("\nPer-Class Breakdown (All 60 Samples):")
    print(f"{'Class':<28} | {'Total':<6} | {'Correct':<8} | {'Kev Acc':<8} | {'Pred Count':<10}")
    print("-" * 68)
    for c in classes:
        c_total = sum(1 for r in records if r["expected_emotion"] == c)
        c_corr = sum(1 for r in records if r["expected_emotion"] == c and r["match"])
        c_pred = sum(1 for r in records if r["predicted_emotion"] == c)
        acc_str = f"{c_corr/c_total:.1%}" if c_total > 0 else "N/A"
        print(f"{c:<28} | {c_total:<6} | {c_corr:<8} | {acc_str:<8} | {c_pred:<10}")

    # Save artifact
    output_path = "benchmark_kev_0.8b_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "jaredpalmer/kev-0.8b",
            "overall_accuracy": all_acc,
            "english_accuracy": en_acc,
            "japanese_accuracy": ja_acc,
            "average_latency_ms": avg_lat,
            "ser_contradiction_rate": contradictions / len(records),
            "records": records
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved detailed evaluation results to {output_path}")

if __name__ == "__main__":
    main()
