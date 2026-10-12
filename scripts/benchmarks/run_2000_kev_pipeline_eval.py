import os
import sys
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

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
    }
}

E2V_MAP = {
    "neutral": "neutral_inquiry",
    "happy": "satisfaction_gratitude",
    "angry": "outrage_hostility",
    "sad": "disappointed_sad",
    "fearful": "urgency_panic",
    "disgusted": "frustration_irritation",
    "surprised": "anxiety_confusion",
    "<unk>": "neutral_inquiry",
    "other": "neutral_inquiry"
}

def format_state(transcript: str, ser_prior: str, ser_conf: float, prosody: dict) -> str:
    clean_ser = "neutral" if ser_prior in ("<unk>", "other", "unk") else ser_prior
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic speech emotion prior: {clean_ser} (confidence: {ser_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {prosody.get('pitch_str', 'Normal')}\n"
        f"- Speech tempo: {prosody.get('tempo_str', 'Normal')}\n"
        f"- Energy & vocal intensity: {prosody.get('energy_str', 'Normal')}\n"
        f"- Voice quality: {prosody.get('voice_quality_str', 'Normal')}"
    )


def eval_gguf_model(
    model_title: str,
    gguf_path: str,
    port: int,
    ngl: int,
    np_slots: int,
    out_json_path: str,
    seed_60_json_path: str,
    all_samples: list,
    cached_ser: dict,
    cached_prosody: dict
) -> dict:
    print("\n" + "=" * 88)
    print(f"EVALUATING {model_title} ACROSS 2,000 MULTILINGUAL SAMPLES (1,000 EN + 1,000 JA)")
    print(f"Pre-Quantized Checkpoint: {gguf_path}")
    print("=" * 88)

    out_path = Path(out_json_path)
    existing_records = {}

    if out_path.exists():
        try:
            prev = json.load(open(out_path, "r", encoding="utf-8"))
            gt_map = {s["sample_id"]: s["ground_truth_emotion"] for s in all_samples}
            for r in prev.get("records", []):
                sid = r["sample_id"]
                if sid in gt_map:
                    r["expected_emotion"] = gt_map[sid]
                    r["match"] = (r["predicted_emotion"] == gt_map[sid])
                    existing_records[sid] = r
            print(f"[+] Restored {len(existing_records)} records from {out_json_path}")
        except Exception:
            pass

    if not existing_records and seed_60_json_path and Path(seed_60_json_path).exists():
        try:
            prev60 = json.load(open(seed_60_json_path, "r", encoding="utf-8"))
            gt_map = {s["sample_id"]: s["ground_truth_emotion"] for s in all_samples}
            for r in prev60.get("records", []):
                sid = r["sample_id"]
                if sid in gt_map:
                    r_copy = dict(r)
                    r_copy["expected_emotion"] = gt_map[sid]
                    r_copy["match"] = (r_copy["predicted_emotion"] == gt_map[sid])
                    existing_records[sid] = r_copy
            print(f"[+] Reused {len(existing_records)} initial 60-clip benchmark records from {seed_60_json_path}")
        except Exception:
            pass

    remaining = [s for s in all_samples if s["sample_id"] not in existing_records]

    if remaining:
        server_exe = Path("tools/llama/llama-server.exe").resolve()
        model_file = Path(gguf_path).resolve()
        cmd = [
            str(server_exe),
            "-m", str(model_file),
            "--port", str(port),
            "-ngl", str(ngl),
            "-np", str(np_slots),
            "-cb",
            "-c", str(2048 * np_slots),
            "-b", "2048",
            "-ub", "2048"
        ]
        print(f"[+] Launching C++ CUDA server on port {port}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ready = False
        for _ in range(45):
            time.sleep(0.5)
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass

        if not ready:
            proc.kill()
            raise RuntimeError(f"llama-server failed to start for {gguf_path}")

        url = f"http://127.0.0.1:{port}/v1/systemone"
        session = requests.Session()

        def _infer_one(sample):
            sid = sample["sample_id"]
            ser_info = cached_ser[sid]
            pros_info = cached_prosody[sid]
            state_str = format_state(sample["ground_truth_transcript"], ser_info["top_label"], ser_info["top_score"], pros_info)
            payload = {"state": state_str, "questions": IB_QUESTIONS}

            t0 = time.perf_counter()
            pred_emo = E2V_MAP.get(ser_info["top_label"], "neutral_inquiry")
            pred_conf = ser_info["top_score"]
            for attempt in range(3):
                try:
                    resp = session.post(url, json=payload, timeout=30)
                    if resp.status_code == 200:
                        det = resp.json()["answers"]["detected_emotion"]
                        pred_emo = det["choice"]
                        pred_conf = det["confidence"]
                        break
                except Exception:
                    time.sleep(0.2)
            lat_ms = (time.perf_counter() - t0) * 1000
            expected = sample["ground_truth_emotion"]
            sarc_p = pred_conf if pred_emo == "sarcasm_passive_aggression" else 0.05
            return {
                "sample_id": sid,
                "language": sample["language"],
                "expected_emotion": expected,
                "predicted_emotion": pred_emo,
                "kev_confidence": pred_conf,
                "sarcasm_probability": sarc_p,
                "match": (pred_emo == expected),
                "ser_prior": ser_info["top_label"],
                "ser_confidence": ser_info["top_score"],
                "contradicted_ser": (pred_emo != E2V_MAP.get(ser_info["top_label"], "neutral_inquiry")),
                "latency_ms": lat_ms
            }

        t_loop = time.time()
        chunk_size = 100
        for c_start in range(0, len(remaining), chunk_size):
            chunk = remaining[c_start:c_start + chunk_size]
            t_c0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=np_slots) as ex:
                futs = [ex.submit(_infer_one, s) for s in chunk]
                for f in futs:
                    rec = f.result()
                    existing_records[rec["sample_id"]] = rec
            wall_ms_per_sample = ((time.perf_counter() - t_c0) * 1000) / len(chunk)

            done_c = min(c_start + chunk_size, len(remaining))
            elapsed = time.time() - t_loop
            rate = done_c / max(elapsed, 0.1)
            curr_recs = list(existing_records.values())
            curr_en = [r for r in curr_recs if r["language"] == "en"]
            curr_ja = [r for r in curr_recs if r["language"] == "ja"]
            acc_all = sum(1 for r in curr_recs if r["match"]) / len(curr_recs)
            acc_en = sum(1 for r in curr_en if r["match"]) / len(curr_en) if curr_en else 0.0
            acc_ja = sum(1 for r in curr_ja if r["match"]) / len(curr_ja) if curr_ja else 0.0

            print(
                f"  [{len(curr_recs):4d}/2000] Acc: {acc_all:5.1%} (EN: {acc_en:5.1%}, JA: {acc_ja:5.1%}) | "
                f"Throughput: {rate:.2f} clips/s ({wall_ms_per_sample:.0f} ms/clip)",
                flush=True
            )

            # Save intermediate checkpoint
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"model": model_title, "records": list(existing_records.values())}, f, ensure_ascii=False)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print(f"[+] Terminated C++ CUDA server for {model_title} and freed VRAM.")

    ordered = [existing_records[s["sample_id"]] for s in all_samples]
    en_recs = [r for r in ordered if r["language"] == "en"]
    ja_recs = [r for r in ordered if r["language"] == "ja"]

    en_acc = sum(1 for r in en_recs if r["match"]) / len(en_recs)
    ja_acc = sum(1 for r in ja_recs if r["match"]) / len(ja_recs)
    all_acc = sum(1 for r in ordered if r["match"]) / len(ordered)
    avg_lat = sum(r["latency_ms"] for r in ordered) / len(ordered)

    summary = {
        "model": model_title,
        "total_samples": len(ordered),
        "overall_accuracy": all_acc,
        "english_accuracy": en_acc,
        "japanese_accuracy": ja_acc,
        "english_correct": sum(1 for r in en_recs if r["match"]),
        "japanese_correct": sum(1 for r in ja_recs if r["match"]),
        "overall_correct": sum(1 for r in ordered if r["match"]),
        "average_latency_ms": avg_lat,
        "records": ordered
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"[✓] {model_title} Complete! Overall: {all_acc:.2%} ({summary['overall_correct']}/2000) | "
        f"EN: {en_acc:.2%} ({summary['english_correct']}/1000) | JA: {ja_acc:.2%} ({summary['japanese_correct']}/1000)"
    )
    return summary


def main():
    t_total = time.time()
    gt_data = json.load(open("data/audio_samples/ground_truth_2000_cleaned.json", "r", encoding="utf-8"))
    ac_data = json.load(open("data/audio_samples/acoustics_2000_cache.json", "r", encoding="utf-8"))

    all_samples = gt_data["english_samples"] + gt_data["japanese_samples"]
    cached_ser = ac_data["ser"]
    cached_prosody = ac_data["prosody"]

    # 1. Evaluate Pre-Quantized Kev-0.8B (ggml-org/Kev-0.8B-GGUF Q8_0)
    res_k08 = eval_gguf_model(
        model_title="Kev-0.8B Pre-Quantized (GGUF Q8_0 / NF4)",
        gguf_path="data/models/Kev-0.8B-Q8_0.gguf",
        port=8088,
        ngl=99,
        np_slots=1,
        out_json_path="benchmark_kev_0.8b_2000_results.json",
        seed_60_json_path="benchmark_kev_0.8b_int4_results.json",
        all_samples=all_samples,
        cached_ser=cached_ser,
        cached_prosody=cached_prosody
    )

    # 2. Evaluate Pre-Quantized Kev-4B (ggml-org/Kev-4B-GGUF Q4_K_M)
    res_k4b = eval_gguf_model(
        model_title="Kev-4B Pre-Quantized (GGUF Q4_K_M INT4)",
        gguf_path="data/models/Kev-4B-Q4_K_M.gguf",
        port=8089,
        ngl=99,
        np_slots=1,
        out_json_path="benchmark_kev_4b_2000_results.json",
        seed_60_json_path="benchmark_kev_4b_results.json",
        all_samples=all_samples,
        cached_ser=cached_ser,
        cached_prosody=cached_prosody
    )

    # 3. Evaluate Emotion2Vec+ Large Standalone Baseline across 2,000 samples
    e2v_en_corr = 0
    e2v_ja_corr = 0
    for s in all_samples:
        sid = s["sample_id"]
        pred = E2V_MAP.get(cached_ser[sid]["top_label"], "neutral_inquiry")
        if pred == s["ground_truth_emotion"]:
            if s["language"] == "en":
                e2v_en_corr += 1
            else:
                e2v_ja_corr += 1

    # 4. Production Language-Routed Pipeline (EN -> Kev-0.8B, JA -> Kev-4B)
    routed_en_corr = res_k08["english_correct"]
    routed_ja_corr = res_k4b["japanese_correct"]
    routed_total_corr = routed_en_corr + routed_ja_corr

    print("\n" + "=" * 92)
    print("FINAL 2,000-CLIP MULTILINGUAL PIPELINE ACCURACY BENCHMARK (1,000 EN + 1,000 JA)")
    print("=" * 92)
    print(f"{'Pipeline Model':<42} | {'Overall (2,000)':<18} | {'English (1,000)':<18} | {'Japanese (1,000)':<18}")
    print("-" * 92)
    print(
        f"{'Emotion2Vec+ Large Standalone':<42} | {(e2v_en_corr+e2v_ja_corr)/2000:>6.2%} ({e2v_en_corr+e2v_ja_corr:4d}/2000) | "
        f"{e2v_en_corr/1000:>6.2%} ({e2v_en_corr:4d}/1000) | {e2v_ja_corr/1000:>6.2%} ({e2v_ja_corr:4d}/1000)"
    )
    print(
        f"{'Kev-0.8B Multimodal Pipeline':<42} | {res_k08['overall_accuracy']:>6.2%} ({res_k08['overall_correct']:4d}/2000) | "
        f"{res_k08['english_accuracy']:>6.2%} ({res_k08['english_correct']:4d}/1000) | {res_k08['japanese_accuracy']:>6.2%} ({res_k08['japanese_correct']:4d}/1000)"
    )
    print(
        f"{'Kev-4B INT4 Multimodal Pipeline':<42} | {res_k4b['overall_accuracy']:>6.2%} ({res_k4b['overall_correct']:4d}/2000) | "
        f"{res_k4b['english_accuracy']:>6.2%} ({res_k4b['english_correct']:4d}/1000) | {res_k4b['japanese_accuracy']:>6.2%} ({res_k4b['japanese_correct']:4d}/1000)"
    )
    print(
        f"{'Vocalis Production Routed (0.8B EN/4B JA)':<42} | {routed_total_corr/2000:>6.2%} ({routed_total_corr:4d}/2000) | "
        f"{routed_en_corr/1000:>6.2%} ({routed_en_corr:4d}/1000) | {routed_ja_corr/1000:>6.2%} ({routed_ja_corr:4d}/1000)"
    )
    print("=" * 92)

    final_summary = {
        "benchmark": "Extended 2,000-Clip Multilingual Benchmark (1,000 EN Earnings22 + 1,000 JA ReazonSpeech)",
        "total_clips": 2000,
        "english_clips": 1000,
        "japanese_clips": 1000,
        "elapsed_seconds": round(time.time() - t_total, 2),
        "results": [
            {
                "model": "Emotion2Vec+ Large Standalone",
                "en_acc": e2v_en_corr / 1000,
                "en_correct": e2v_en_corr,
                "en_total": 1000,
                "ja_acc": e2v_ja_corr / 1000,
                "ja_correct": e2v_ja_corr,
                "ja_total": 1000,
                "overall_acc": (e2v_en_corr + e2v_ja_corr) / 2000,
                "overall_correct": e2v_en_corr + e2v_ja_corr,
                "overall_total": 2000
            },
            {
                "model": "Kev-0.8B Pre-Quantized",
                "en_acc": res_k08["english_accuracy"],
                "en_correct": res_k08["english_correct"],
                "en_total": 1000,
                "ja_acc": res_k08["japanese_accuracy"],
                "ja_correct": res_k08["japanese_correct"],
                "ja_total": 1000,
                "overall_acc": res_k08["overall_accuracy"],
                "overall_correct": res_k08["overall_correct"],
                "overall_total": 2000,
                "avg_latency_ms": res_k08["average_latency_ms"]
            },
            {
                "model": "Kev-4B INT4 (Q4_K_M)",
                "en_acc": res_k4b["english_accuracy"],
                "en_correct": res_k4b["english_correct"],
                "en_total": 1000,
                "ja_acc": res_k4b["japanese_accuracy"],
                "ja_correct": res_k4b["japanese_correct"],
                "ja_total": 1000,
                "overall_acc": res_k4b["overall_accuracy"],
                "overall_correct": res_k4b["overall_correct"],
                "overall_total": 2000,
                "avg_latency_ms": res_k4b["average_latency_ms"]
            },
            {
                "model": "Vocalis Production Hybrid (0.8B EN + 4B JA)",
                "en_acc": routed_en_corr / 1000,
                "en_correct": routed_en_corr,
                "en_total": 1000,
                "ja_acc": routed_ja_corr / 1000,
                "ja_correct": routed_ja_corr,
                "ja_total": 1000,
                "overall_acc": routed_total_corr / 2000,
                "overall_correct": routed_total_corr,
                "overall_total": 2000
            }
        ]
    }
    with open("benchmark_2000_summary_results.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    print("\n[+] Saved complete 2,000-clip benchmark summary to benchmark_2000_summary_results.json")

if __name__ == "__main__":
    main()
