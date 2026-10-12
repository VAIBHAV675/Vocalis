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

# Environment config
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

import torch
from extract_prosody import extract_prosody_features

# 8-EMOTION QUESTION SCHEMA FOR INVESTMENT BANKING ARBITRATION
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the speaker transcript alongside acoustic prosody and Speech Emotion Recognition (SER) cues. "
            "Resolve any contradictions between spoken words and vocal acoustics (e.g. sarcasm, masked anger). "
            "Classify into one of the 8 canonical emotions based on the customer's true mental and emotional state."
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

def format_laya_prompt(transcript: str, ser_prior: str, ser_conf: float, prosody: dict) -> str:
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic speech emotion prior: {ser_prior} (confidence: {ser_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {prosody['pitch_str']}\n"
        f"- Speech tempo: {prosody['tempo_str']}\n"
        f"- Energy & vocal intensity: {prosody['energy_str']}\n"
        f"- Voice quality: {prosody['voice_quality_str']}"
    )

def main():
    print("=" * 80)
    print("TRI-MODAL SPEECH EMOTION RECOGNITION BENCHMARK (60 REAL AUDIO RECORDINGS)")
    print("MODELS: Emotion2Vec+ Large (SER) + Librosa (Prosody) + Laya mmBERT (Arbitration)")
    print(f"CUDA DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # 1. Load Dataset
    gt_path = Path("data/audio_samples/ground_truth_60_multilingual.json")
    if not gt_path.exists():
        print(f"Error: {gt_path} not found.")
        return
    with open(gt_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_samples = dataset["english_samples"] + dataset["japanese_samples"]
    print(f"[+] Loaded {len(all_samples)} total audio clips ({len(dataset['english_samples'])} EN, {len(dataset['japanese_samples'])} JA)")

    # 2. Extract Librosa Prosody Features for all 60 samples
    print("\n--- STEP 1: EXTRACTING ACOUSTIC PROSODY FEATURES (LIBROSA) ---")
    prosody_results = {}
    prosody_latencies = []
    for s in all_samples:
        sid = s["sample_id"]
        wav_path = s["wav_path"]
        t0 = time.perf_counter()
        features = extract_prosody_features(wav_path)
        lat = (time.perf_counter() - t0) * 1000
        features["latency_ms"] = lat
        prosody_latencies.append(lat)
        prosody_results[sid] = features
    print(f"[+] Librosa Prosody Extracted for 60 files in {sum(prosody_latencies)/1000:.2f}s (avg: {sum(prosody_latencies)/len(prosody_latencies):.1f} ms/sample)")

    # 3. Load Emotion2Vec+ Large
    print("\n--- STEP 2: RUNNING EMOTION2VEC+ LARGE SER ACOUSTIC INFERENCE ---")
    from funasr import AutoModel
    t0 = time.time()
    # Use GPU if available or CPU
    e2v_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    e2v_model = AutoModel(model="iic/emotion2vec_plus_large", hub="ms", device=e2v_device)
    print(f"[+] Emotion2Vec+ Large loaded in {time.time() - t0:.2f}s on {e2v_device}")

    ser_results = {}
    ser_latencies = []
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)

    for idx, s in enumerate(all_samples, 1):
        sid = s["sample_id"]
        wav_path = s["wav_path"]
        t_start = time.perf_counter()
        try:
            res = e2v_model.generate(wav_path, output_dir=str(scratch_dir), granularity="utterance", extract_embedding=False)
            # Parse emotion2vec result
            import numpy as np
            if isinstance(res, list) and len(res) > 0:
                item = res[0]
                labels = item.get("labels", [])
                scores = item.get("scores", [])
                if labels and scores:
                    best_idx = int(np.argmax(scores))
                    raw_label = labels[best_idx]
                    top_label = raw_label.split("/")[-1].strip()
                    top_score = float(scores[best_idx])
                else:
                    top_label = "neutral"
                    top_score = 0.5
            elif isinstance(res, dict):
                labels = res.get("labels", [])
                scores = res.get("scores", [])
                if labels and scores:
                    best_idx = int(np.argmax(scores))
                    raw_label = labels[best_idx]
                    top_label = raw_label.split("/")[-1].strip()
                    top_score = float(scores[best_idx])
                else:
                    top_label = "neutral"
                    top_score = 0.5
            else:
                top_label = "neutral"
                top_score = 0.5
        except Exception as e:
            top_label = "neutral"
            top_score = 0.5

        lat = (time.perf_counter() - t_start) * 1000
        ser_latencies.append(lat)
        ser_results[sid] = {
            "top_label": top_label,
            "top_score": top_score,
            "latency_ms": lat
        }
        if idx % 10 == 0 or idx == len(all_samples):
            print(f"  [{idx}/{len(all_samples)}] Processed SER -> {sid}: {top_label} ({top_score:.1%}, {lat:.1f}ms)")

    print(f"[+] Emotion2Vec+ Large SER completed (avg: {sum(ser_latencies)/len(ser_latencies):.1f} ms/sample)")

    # Flush Emotion2Vec from GPU memory so Laya has 100% of VRAM
    print("\n[+] Unloading Emotion2Vec+ Large from GPU memory to free VRAM for Laya mmBERT...")
    del e2v_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    # 4. Load Laya mmBERT Router
    print("\n--- STEP 3: LAYA mmBERT NATURAL LANGUAGE ARBITRATION (CUDA) ---")
    from laya import Router
    t0 = time.time()
    router = Router(preload=False)
    router.preload(["multilingual"])
    print(f"[+] Laya mmBERT preloaded in {time.time() - t0:.2f}s\n")

    # 5. Evaluate on all 60 samples
    print(f"{'ID':<16} | {'Lang':<4} | {'Expected':<22} | {'Predicted':<22} | {'Conf':<6} | {'Match'}")
    print("-" * 88)

    evaluation_records = []
    laya_latencies = []

    for s in all_samples:
        sid = s["sample_id"]
        lang = s["language"]
        expected = s["ground_truth_emotion"]
        transcript = s["ground_truth_transcript"]
        ser_info = ser_results[sid]
        prosody_info = prosody_results[sid]

        prompt = format_laya_prompt(
            transcript=transcript,
            ser_prior=ser_info["top_label"],
            ser_conf=ser_info["top_score"],
            prosody=prosody_info
        )

        t_start = time.perf_counter()
        with torch.inference_mode():
            res = router.predict(prompt, IB_QUESTIONS, model="multilingual")
        lat = (time.perf_counter() - t_start) * 1000
        laya_latencies.append(lat)

        pred_emo = res["answers"]["detected_emotion"]["choice"]
        pred_conf = res["answers"]["detected_emotion"]["answer_confidence"]
        sarc_p = res["answers"]["sarcasm_detected"]["noul"]
        match = (pred_emo == expected)

        match_str = "[PASS]" if match else "[FAIL]"
        print(f"{sid:<16} | {lang:<4} | {expected:<22} | {pred_emo:<22} | {pred_conf:>5.1%} | {match_str}")

        evaluation_records.append({
            "sample_id": sid,
            "language": lang,
            "filename": s["filename"],
            "expected_emotion": expected,
            "predicted_emotion": pred_emo,
            "laya_confidence": pred_conf,
            "sarcasm_probability": sarc_p,
            "match": match,
            "ser_prior": ser_info["top_label"],
            "ser_confidence": ser_info["top_score"],
            "prosody": prosody_info,
            "latencies": {
                "prosody_ms": prosody_info.get("latency_ms", 0.0),
                "ser_ms": ser_info["latency_ms"],
                "laya_ms": lat,
                "total_ms": prosody_info.get("latency_ms", 0.0) + ser_info["latency_ms"] + lat
            }
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 6. Metrics Summary
    en_records = [r for r in evaluation_records if r["language"] == "en"]
    ja_records = [r for r in evaluation_records if r["language"] == "ja"]

    en_acc = sum(1 for r in en_records if r["match"]) / len(en_records)
    ja_acc = sum(1 for r in ja_records if r["match"]) / len(ja_records)
    total_acc = sum(1 for r in evaluation_records if r["match"]) / len(evaluation_records)

    avg_prosody_lat = sum(prosody_latencies) / len(prosody_latencies)
    avg_ser_lat = sum(ser_latencies) / len(ser_latencies)
    avg_laya_lat = sum(laya_latencies) / len(laya_latencies)
    avg_total_lat = avg_prosody_lat + avg_ser_lat + avg_laya_lat

    print("\n" + "=" * 80)
    print("FINAL 60-AUDIO PIPELINE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Dataset Audio Files : {len(evaluation_records)} (30 EN Earnings22, 30 JA ReazonSpeech)")
    print(f"Overall Accuracy Across 8 Emotions : {total_acc:.1%} ({sum(1 for r in evaluation_records if r['match'])}/60)")
    print(f"English-Only Accuracy (30 Calls)   : {en_acc:.1%} ({sum(1 for r in en_records if r['match'])}/30)")
    print(f"Japanese-Only Accuracy (30 Calls)  : {ja_acc:.1%} ({sum(1 for r in ja_records if r['match'])}/30)")
    print("-" * 80)
    print("PIPELINE LATENCY BREAKDOWN (Average per audio call):")
    print(f"  1. Librosa Prosody Extraction : {avg_prosody_lat:>6.1f} ms")
    print(f"  2. Emotion2Vec+ Large SER     : {avg_ser_lat:>6.1f} ms")
    print(f"  3. Laya mmBERT Arbitration    : {avg_laya_lat:>6.1f} ms")
    print(f"  Total Average Pipeline Latency: {avg_total_lat:>6.1f} ms (~{avg_total_lat/1000:.2f}s)")
    print("=" * 80)

    # Save to JSON
    output_summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(evaluation_records),
        "overall_accuracy": total_acc,
        "english_accuracy": en_acc,
        "japanese_accuracy": ja_acc,
        "latency_breakdown_ms": {
            "prosody_avg_ms": avg_prosody_lat,
            "ser_avg_ms": avg_ser_lat,
            "laya_avg_ms": avg_laya_lat,
            "total_avg_ms": avg_total_lat
        },
        "records": evaluation_records
    }

    with open("benchmark_60_audio_results.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)
    print(f"Full benchmark detailed output saved to benchmark_60_audio_results.json")

if __name__ == "__main__":
    main()
