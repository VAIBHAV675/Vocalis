import os
import sys
import json
import time
import requests
from pathlib import Path

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
    print("KEV-4B INT4 QUANTIZED BENCHMARK (60 REAL AUDIO RECORDINGS)")
    print("Model: ggml-org/Kev-4B-GGUF (Q4_K_M) on CUDA via /v1/systemone")
    print("=" * 80)

    # 1. Load ground truth & baseline results
    with open("benchmark_60_audio_results.json", "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    samples = baseline_data["records"]

    with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    transcripts = {s["sample_id"]: s["ground_truth_transcript"] for s in (gt_data["english_samples"] + gt_data["japanese_samples"])}

    print(f"[+] Loaded {len(samples)} benchmark records (30 EN, 30 JA)")

    # 2. Benchmark Loop
    print(f"\n{'ID':<16} | {'Lang':<4} | {'Expected':<24} | {'Kev-4B Pred':<24} | {'Conf':<6} | {'Match':<6} | {'Lat(ms)':<7}")
    print("-" * 102)

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

        req_payload = {
            "state": state_text,
            "questions": IB_QUESTIONS
        }

        lat = 0
        res_data = None
        for attempt in range(3):
            try:
                t_start = time.perf_counter()
                resp = requests.post("http://127.0.0.1:8089/v1/systemone", json=req_payload, timeout=30)
                lat = (time.perf_counter() - t_start) * 1000
                res_data = resp.json()["answers"]
                break
            except Exception as e:
                time.sleep(1)
                if attempt == 2:
                    raise e
        latencies.append(lat)
        pred_emo = res_data["detected_emotion"]["choice"]
        pred_conf = res_data["detected_emotion"]["confidence"]
        sarc_p = res_data["sarcasm_detected"]["noul"]

        match = (pred_emo == expected)
        match_str = "[PASS]" if match else "[FAIL]"

        contradicted_ser = (pred_emo != ser_prior)
        if contradicted_ser:
            contradictions += 1

        print(f"{sid:<16} | {lang:<4} | {expected:<24} | {pred_emo:<24} | {pred_conf:>5.1%} | {match_str:<6} | {lat:>6.1f}ms", flush=True)

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
            "latency_ms": lat
        })

    # Summary Metrics
    en_records = [r for r in records if r["language"] == "en"]
    ja_records = [r for r in records if r["language"] == "ja"]

    en_acc = sum(1 for r in en_records if r["match"]) / len(en_records)
    ja_acc = sum(1 for r in ja_records if r["match"]) / len(ja_records)
    all_acc = sum(1 for r in records if r["match"]) / len(records)
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY: KEV-4B vs KEV-0.8B vs LAYA mmBERT")
    print("=" * 80)
    print(f"Overall Accuracy:  Kev-4B: {all_acc:.1%} ({sum(r['match'] for r in records)}/{len(records)}) | Kev-0.8B: 58.3% | Laya: 43.3%")
    print(f"English Accuracy:  Kev-4B: {en_acc:.1%} ({sum(r['match'] for r in en_records)}/{len(en_records)}) | Kev-0.8B: 70.0% | Laya: 50.0%")
    print(f"Japanese Accuracy: Kev-4B: {ja_acc:.1%} ({sum(r['match'] for r in ja_records)}/{len(ja_records)}) | Kev-0.8B: 46.7% | Laya: 36.7%")
    print(f"Average Latency:   Kev-4B: {avg_lat:.1f}ms | Kev-0.8B: ~2360ms | Laya: ~1200ms")
    print("=" * 80)

    # Class Breakdown
    classes = list(IB_QUESTIONS["detected_emotion"]["criteria"].keys())
    print("\nPer-Class Breakdown (Kev-4B on All 60 Samples):")
    print(f"{'Class':<28} | {'Total':<6} | {'Correct':<8} | {'Kev-4B Acc':<10} | {'Pred Count':<10}")
    print("-" * 70)
    for c in classes:
        c_total = sum(1 for r in records if r["expected_emotion"] == c)
        c_corr = sum(1 for r in records if r["expected_emotion"] == c and r["match"])
        c_pred = sum(1 for r in records if r["predicted_emotion"] == c)
        acc_str = f"{c_corr/c_total:.1%}" if c_total > 0 else "N/A"
        print(f"{c:<28} | {c_total:<6} | {c_corr:<8} | {acc_str:<10} | {c_pred:<10}")

    # Save artifact
    output_path = "benchmark_kev_4b_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ggml-org/Kev-4B-GGUF (Q4_K_M)",
            "overall_accuracy": all_acc,
            "english_accuracy": en_acc,
            "japanese_accuracy": ja_acc,
            "average_latency_ms": avg_lat,
            "records": records
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved detailed evaluation results to {output_path}")

if __name__ == "__main__":
    main()
