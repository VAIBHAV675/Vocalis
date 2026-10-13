import os
import sys
import io
import gc
import time
import json
import socket
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Force IPv4 so HuggingFace downloads never hang on IPv6
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download, HfApi
from extract_prosody import extract_prosody_features
from vocalis_segmented_pipeline import (
    IB_QUESTIONS,
    EMOTION_MAP_E2V as E2V_MAP,
    clean_e2v_label,
    format_state as format_kev_prompt,
)

def emotion_to_sentiment(emo_key: str, s_flag: bool = False) -> str:
    if s_flag:
        return "negative"
    if emo_key == "satisfaction_gratitude":
        return "positive"
    if emo_key == "neutral_inquiry":
        return "neutral"
    return "negative"

SER_GT_TO_CANONICAL_SET = {
    "neutral": {"neutral_inquiry"},
    "calm": {"neutral_inquiry"},
    "happiness": {"satisfaction_gratitude"},
    "happy": {"satisfaction_gratitude"},
    "pleasant_surprise": {"satisfaction_gratitude"},
    "sadness": {"disappointed_sad"},
    "sad": {"disappointed_sad"},
    "anger": {"frustration_irritation", "outrage_hostility", "sarcasm_passive_aggression"},
    "angry": {"frustration_irritation", "outrage_hostility", "sarcasm_passive_aggression"},
    "disgust": {"frustration_irritation", "outrage_hostility", "sarcasm_passive_aggression"},
    "fear": {"anxiety_confusion", "urgency_panic"},
}

SER_GT_TO_PRIMARY_CANONICAL = {
    "neutral": "neutral_inquiry",
    "calm": "neutral_inquiry",
    "happiness": "satisfaction_gratitude",
    "happy": "satisfaction_gratitude",
    "pleasant_surprise": "satisfaction_gratitude",
    "sadness": "disappointed_sad",
    "sad": "disappointed_sad",
    "anger": "outrage_hostility",
    "angry": "outrage_hostility",
    "disgust": "frustration_irritation",
    "fear": "anxiety_confusion",
}

SER_GT_TO_SENTIMENT = {
    "neutral": "neutral",
    "calm": "neutral",
    "happiness": "positive",
    "happy": "positive",
    "pleasant_surprise": "positive",
    "sadness": "negative",
    "sad": "negative",
    "anger": "negative",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
}


def prepare_ser_validation_clips(out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    # 1. TESS (Toronto Emotional Speech Set) - 36 stratified clips (6 per emotion)
    tess_pq = hf_hub_download(
        "VibeCheck1/VibeCheckRandomSamples_VibeCheckAudio_TESS",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    tbl_tess = pq.read_table(tess_pq)
    tess_counts = {}
    for i in range(len(tbl_tess)):
        emo = tbl_tess["answer"][i].as_py()
        if emo not in ("neutral", "happiness", "sadness", "angry", "fear", "disgust"):
            continue
        if tess_counts.get(emo, 0) >= 6:
            continue
        tess_counts[emo] = tess_counts.get(emo, 0) + 1
        sid = f"tess_{emo}_{tess_counts[emo]:02d}"
        wav_path = out_dir / f"{sid}.wav"
        if not wav_path.exists():
            audio_bytes = tbl_tess["audio"][i]["bytes"].as_py()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(wav_path), data, sr)
        samples.append({
            "sample_id": sid,
            "dataset": "TESS",
            "language": "en",
            "wav_path": str(wav_path),
            "transcript": tbl_tess["transcription"][i].as_py(),
            "raw_gt_emotion": emo,
            "gt_canonical_set": list(SER_GT_TO_CANONICAL_SET[emo]),
            "gt_primary_emotion": SER_GT_TO_PRIMARY_CANONICAL[emo],
            "gt_sentiment": SER_GT_TO_SENTIMENT[emo],
        })

    # 2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech) - 36 stratified clips (6 per emotion)
    rav_pq = hf_hub_download(
        "VibeCheck1/VibeCheckRandomSamples_VibeCheckAudio_RAVDESS",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    tbl_rav = pq.read_table(rav_pq)
    rav_counts = {}
    for i in range(len(tbl_rav)):
        emo = tbl_rav["answer"][i].as_py()
        bucket = "neutral" if emo in ("neutral", "calm") else emo
        if bucket not in ("neutral", "happiness", "sadness", "anger", "fear", "disgust"):
            continue
        if rav_counts.get(bucket, 0) >= 6:
            continue
        rav_counts[bucket] = rav_counts.get(bucket, 0) + 1
        sid = f"ravdess_{bucket}_{rav_counts[bucket]:02d}"
        wav_path = out_dir / f"{sid}.wav"
        if not wav_path.exists():
            audio_bytes = tbl_rav["audio"][i]["bytes"].as_py()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(wav_path), data, sr)
        samples.append({
            "sample_id": sid,
            "dataset": "RAVDESS",
            "language": "en",
            "wav_path": str(wav_path),
            "transcript": tbl_rav["transcription"][i].as_py(),
            "raw_gt_emotion": emo,
            "gt_canonical_set": list(SER_GT_TO_CANONICAL_SET[emo]),
            "gt_primary_emotion": SER_GT_TO_PRIMARY_CANONICAL[emo],
            "gt_sentiment": SER_GT_TO_SENTIMENT[emo],
        })

    # 3. SAVEE (Surrey Audio-Visual Expressed Emotion) - 25 stratified clips
    savee_pq = hf_hub_download(
        "VibeCheck1/VibeCheckRandomSamples_VibeCheckAudio_SAVEE",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    tbl_savee = pq.read_table(savee_pq)
    savee_counts = {}
    for i in range(len(tbl_savee)):
        emo = tbl_savee["answer"][i].as_py()
        if emo not in ("happiness", "sadness", "anger", "fear", "disgust"):
            continue
        if savee_counts.get(emo, 0) >= 5:
            continue
        savee_counts[emo] = savee_counts.get(emo, 0) + 1
        sid = f"savee_{emo}_{savee_counts[emo]:02d}"
        wav_path = out_dir / f"{sid}.wav"
        if not wav_path.exists():
            audio_bytes = tbl_savee["audio"][i]["bytes"].as_py()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(wav_path), data, sr)
        samples.append({
            "sample_id": sid,
            "dataset": "SAVEE",
            "language": "en",
            "wav_path": str(wav_path),
            "transcript": tbl_savee["transcription"][i].as_py(),
            "raw_gt_emotion": emo,
            "gt_canonical_set": list(SER_GT_TO_CANONICAL_SET[emo]),
            "gt_primary_emotion": SER_GT_TO_PRIMARY_CANONICAL[emo],
            "gt_sentiment": SER_GT_TO_SENTIMENT[emo],
        })

    # 4. JVNV (Japanese Emotional Speech Corpus) - 25 stratified Japanese clips already in out_dir
    for emo in ("anger", "disgust", "fear", "happy", "sad"):
        for idx in range(1, 6):
            sid = f"jvnv_ja_{emo}_{idx:02d}"
            wav_path = out_dir / f"{sid}.wav"
            if wav_path.exists():
                samples.append({
                    "sample_id": sid,
                    "dataset": "JVNV-JA",
                    "language": "ja",
                    "wav_path": str(wav_path),
                    "transcript": "",
                    "raw_gt_emotion": emo,
                    "gt_canonical_set": list(SER_GT_TO_CANONICAL_SET[emo]),
                    "gt_primary_emotion": SER_GT_TO_PRIMARY_CANONICAL[emo],
                    "gt_sentiment": SER_GT_TO_SENTIMENT[emo],
                })

    return samples


def run_kev_eval(model_name: str, gguf_path: Path, port: int, ngl: int, samples: list, ser_cache: dict, prosody_cache: dict):
    server_exe = (ROOT / "tools" / "llama" / "llama-server.exe").resolve()
    cmd = [
        str(server_exe),
        "-m", str(gguf_path.resolve()),
        "--port", str(port),
        "-ngl", str(ngl),
        "-np", "2",
        "-cb",
        "-c", "4096",
        "-b", "1024",
        "-ub", "1024",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ready = False
        for _ in range(50):
            time.sleep(0.5)
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
        if not ready:
            raise RuntimeError(f"llama-server did not start on port {port}")

        url = f"http://127.0.0.1:{port}/v1/systemone"
        session = requests.Session()

        def _infer(s):
            sid = s["sample_id"]
            ser_info = ser_cache[sid]
            pros_info = prosody_cache[sid]
            state_str = format_kev_prompt(
                transcript=s["transcript"],
                ser_prior=ser_info["top_label"],
                ser_conf=ser_info["top_score"],
                prosody=pros_info,
            )
            payload = {"state": state_str, "questions": IB_QUESTIONS}
            t0 = time.perf_counter()
            resp = session.post(url, json=payload, timeout=30)
            lat_ms = (time.perf_counter() - t0) * 1000
            ans = resp.json()["answers"]
            det = ans["detected_emotion"]
            sarc = ans.get("sarcasm_detected", {})
            pred_emo = det["choice"]
            pred_conf = float(det.get("confidence", det.get("answer_confidence", 0.5)))
            sarc_prob = float(sarc.get("noul", [0.0])[0] if isinstance(sarc.get("noul"), list) else sarc.get("noul", 0.0))
            sarc_flag = bool(sarc_prob >= 0.45 or pred_emo == "sarcasm_passive_aggression")
            pred_sent = emotion_to_sentiment(pred_emo, sarc_flag)

            emo_match = pred_emo in set(s["gt_canonical_set"])
            sent_match = (pred_sent == s["gt_sentiment"])
            return {
                "sample_id": sid,
                "dataset": s["dataset"],
                "language": s["language"],
                "raw_gt_emotion": s["raw_gt_emotion"],
                "gt_sentiment": s["gt_sentiment"],
                "predicted_emotion": pred_emo,
                "predicted_sentiment": pred_sent,
                "confidence": round(pred_conf, 4),
                "sarcasm_flag": sarc_flag,
                "emotion_match": emo_match,
                "sentiment_match": sent_match,
                "ser_prior": ser_info["top_label"],
                "ser_confidence": round(ser_info["top_score"], 4),
                "latency_ms": round(lat_ms, 2),
            }

        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(_infer, samples))
        return results
    finally:
        proc.kill()
        proc.wait()


def summarize_results(title: str, records: list) -> dict:
    total = len(records)
    emo_corr = sum(1 for r in records if r["emotion_match"])
    sent_corr = sum(1 for r in records if r["sentiment_match"])
    by_ds = {}
    for ds in sorted(set(r["dataset"] for r in records)):
        sub = [r for r in records if r["dataset"] == ds]
        ec = sum(1 for r in sub if r["emotion_match"])
        sc = sum(1 for r in sub if r["sentiment_match"])
        by_ds[ds] = {
            "total": len(sub),
            "emotion_correct": ec,
            "emotion_accuracy": round(ec / len(sub), 4),
            "sentiment_correct": sc,
            "sentiment_accuracy": round(sc / len(sub), 4),
        }
    return {
        "model": title,
        "total": total,
        "emotion_correct": emo_corr,
        "emotion_accuracy": round(emo_corr / total, 4),
        "sentiment_correct": sent_corr,
        "sentiment_accuracy": round(sent_corr / total, 4),
        "by_dataset": by_ds,
    }


def extract_acoustics_stage(samples: list, cache_path: Path):
    import torch
    ja_samples = [s for s in samples if s["language"] == "ja" and not s["transcript"]]
    if ja_samples:
        print(f"[+] Stage 1: Transcribing {len(ja_samples)} JVNV Japanese clips with Faster-Whisper Turbo INT8...", flush=True)
        from faster_whisper import WhisperModel
        w_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="int8_float16")
        for s in ja_samples:
            segs, _ = w_model.transcribe(s["wav_path"], language="ja", beam_size=1)
            s["transcript"] = "".join(seg.text for seg in segs).strip() or "（発話）"
        del w_model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[+] Stage 2A: Extracting Librosa Prosody for {len(samples)} clips...", flush=True)
    prosody_cache = {}
    for s in samples:
        prosody_cache[s["sample_id"]] = extract_prosody_features(s["wav_path"])

    print(f"[+] Stage 2B: Running Emotion2Vec+ Large SER on {len(samples)} clips...", flush=True)
    from funasr import AutoModel
    e2v = AutoModel(model="iic/emotion2vec_plus_large", hub="ms", device="cuda:0", disable_update=True)
    ser_cache = {}
    scratch_dir = ROOT / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    for s in samples:
        sid = s["sample_id"]
        t0 = time.perf_counter()
        res = e2v.generate(s["wav_path"], output_dir=str(scratch_dir), granularity="utterance", extract_embedding=False)
        lat_ms = (time.perf_counter() - t0) * 1000
        item = res[0] if isinstance(res, list) and res else (res if isinstance(res, dict) else {})
        labels = item.get("labels", [])
        scores = item.get("scores", [])
        if labels and scores:
            best_idx = int(np.argmax(scores))
            top_label = clean_e2v_label(labels[best_idx])
            top_score = float(scores[best_idx])
        else:
            top_label, top_score = "neutral", 0.5
        ser_cache[sid] = {"top_label": top_label, "top_score": top_score, "latency_ms": lat_ms}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "transcripts": {s["sample_id"]: s["transcript"] for s in samples},
            "ser_cache": ser_cache,
            "prosody_cache": prosody_cache,
        }, f, indent=2, ensure_ascii=False)


def main():
    t_all = time.time()
    out_dir = ROOT / "data" / "ser_validation_clips"
    cache_path = ROOT / "scratch" / "ser_validation_acoustics_cache.json"
    samples = prepare_ser_validation_clips(out_dir)

    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        extract_acoustics_stage(samples, cache_path)
        return

    print("=" * 88, flush=True)
    print("VOCALIS EVALUATION ON ACTUAL SER VALIDATION DATASETS (TESS + RAVDESS + SAVEE + JVNV)", flush=True)
    print("=" * 88, flush=True)

    if not cache_path.exists():
        subprocess.run([sys.executable, __file__, "--extract-only"], check=True)

    cached = json.load(open(cache_path, "r", encoding="utf-8"))
    for s in samples:
        s["transcript"] = cached["transcripts"].get(s["sample_id"], s["transcript"])
    ser_cache = cached["ser_cache"]
    prosody_cache = cached["prosody_cache"]

    e2v_records = []
    for s in samples:
        sid = s["sample_id"]
        top_label = ser_cache[sid]["top_label"]
        e2v_canon = E2V_MAP.get(top_label, "neutral_inquiry")
        e2v_sent = emotion_to_sentiment(e2v_canon, False)
        e2v_records.append({
            "sample_id": sid,
            "dataset": s["dataset"],
            "language": s["language"],
            "raw_gt_emotion": s["raw_gt_emotion"],
            "gt_sentiment": s["gt_sentiment"],
            "predicted_emotion": e2v_canon,
            "predicted_sentiment": e2v_sent,
            "emotion_match": e2v_canon in set(s["gt_canonical_set"]),
            "sentiment_match": (e2v_sent == s["gt_sentiment"]),
        })

    print(f"[+] Stage 3A: Running Kev-0.8B (Q8_0) Tri-Modal Arbitration on {len(samples)} clips...", flush=True)
    kev08_records = run_kev_eval(
        "Kev-0.8B (Q8_0)",
        ROOT / "data" / "models" / "Kev-0.8B-Q8_0.gguf",
        8191,
        99,
        samples,
        ser_cache,
        prosody_cache,
    )

    print(f"[+] Stage 3B: Running Kev-4B (INT4 Q4_K_M) Tri-Modal Arbitration on {len(samples)} clips...", flush=True)
    kev4b_records = run_kev_eval(
        "Kev-4B (INT4 Q4_K_M)",
        ROOT / "data" / "models" / "Kev-4B-Q4_K_M.gguf",
        8192,
        28,
        samples,
        ser_cache,
        prosody_cache,
    )

    e2v_sum = summarize_results("Emotion2Vec+ Large (Standalone SER)", e2v_records)
    k08_sum = summarize_results("Vocalis + Kev-0.8B (Q8_0)", kev08_records)
    k4b_sum = summarize_results("Vocalis + Kev-4B (INT4 Q4_K_M)", kev4b_records)

    out_payload = {
        "benchmark": "Multi-Corpus SER Validation Benchmark (TESS + RAVDESS + SAVEE + JVNV-JA)",
        "total_clips": len(samples),
        "elapsed_sec": round(time.time() - t_all, 2),
        "summaries": {
            "emotion2vec_plus_large": e2v_sum,
            "vocalis_kev_0.8b": k08_sum,
            "vocalis_kev_4b_int4": k4b_sum,
        },
        "kev_4b_records": kev4b_records,
        "kev_0.8b_records": kev08_records,
    }

    out_path = ROOT / "benchmarks" / "results" / "ser_validation_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88, flush=True)
    print(f"SER VALIDATION BENCHMARK COMPLETE IN {out_payload['elapsed_sec']}s ({len(samples)} CLIPS)", flush=True)
    print("=" * 88, flush=True)
    for key, s in out_payload["summaries"].items():
        print(f"\n>>> {s['model']}:", flush=True)
        print(f"    Overall 8-Emotion Accuracy : {s['emotion_correct']}/{s['total']} ({s['emotion_accuracy']*100:.2f}%)", flush=True)
        print(f"    Overall 3-Sentiment Accuracy: {s['sentiment_correct']}/{s['total']} ({s['sentiment_accuracy']*100:.2f}%)", flush=True)
        for ds, dstat in s["by_dataset"].items():
            print(f"      - {ds:<10}: Emotion {dstat['emotion_correct']}/{dstat['total']} ({dstat['emotion_accuracy']*100:.1f}%) | Sentiment {dstat['sentiment_correct']}/{dstat['total']} ({dstat['sentiment_accuracy']*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
