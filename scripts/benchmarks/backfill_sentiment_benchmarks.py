import json
from pathlib import Path


def emo_to_sentiment(emo: str, sarcasm_flag: bool = False) -> str:
    """
    Maps the 8 canonical Vocalis emotions onto 3-way Sentiment Polarity:
    - positive: satisfaction_gratitude (unless sarcasm_flag is True)
    - neutral: neutral_inquiry
    - negative: all 6 negative/escalation classes, OR satisfaction_gratitude with sarcasm_flag=True
    """
    if sarcasm_flag:
        return "negative"
    if emo == "satisfaction_gratitude":
        return "positive"
    if emo == "neutral_inquiry":
        return "neutral"
    return "negative"


def probs_to_sentiment_probs(probs: dict, sarcasm_prob: float = 0.0, sarcasm_flag: bool = False) -> dict:
    p_sat = float(probs.get("satisfaction_gratitude", 0.0))
    p_neu = float(probs.get("neutral_inquiry", 0.0))
    neg_keys = [
        "frustration_irritation",
        "outrage_hostility",
        "disappointed_sad",
        "sarcasm_passive_aggression",
        "anxiety_confusion",
        "urgency_panic",
    ]
    p_neg = sum(float(probs.get(k, 0.0)) for k in neg_keys)
    if sarcasm_flag or float(sarcasm_prob) > 0.45:
        shift = p_sat * max(float(sarcasm_prob), 0.5)
        p_sat -= shift
        p_neg += shift
    tot = p_sat + p_neu + p_neg
    if tot > 0:
        return {
            "positive": round(p_sat / tot, 4),
            "neutral": round(p_neu / tot, 4),
            "negative": round(p_neg / tot, 4),
        }
    return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


def backfill_pipeline_json(fpath: Path):
    if not fpath.exists():
        return
    data = json.loads(fpath.read_text(encoding="utf-8"))
    if "segments" not in data:
        return

    sent_dur = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    sent_prob_accum = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
    tot_dur = 0.0

    for seg in data.get("segments", []):
        s_flag = bool(seg.get("sarcasm_flag", False))
        s_prob = float(seg.get("sarcasm_probability", 0.0))
        pred_sent = emo_to_sentiment(seg.get("predicted_emotion", "neutral_inquiry"), s_flag)
        s_probs = probs_to_sentiment_probs(seg.get("probabilities", {}), s_prob, s_flag)
        seg["predicted_sentiment"] = pred_sent
        seg["sentiment_probabilities"] = s_probs

        dur = float(seg.get("duration_sec", 1.0))
        tot_dur += dur
        sent_dur[pred_sent] += dur
        sent_counts[pred_sent] += 1
        for sk, sv in s_probs.items():
            sent_prob_accum[sk] += sv * dur

    if tot_dur > 0:
        sent_time_pct = {k: round((v / tot_dur) * 100, 2) for k, v in sent_dur.items()}
        global_sent_probs = {k: round(v / tot_dur, 4) for k, v in sent_prob_accum.items()}
    else:
        sent_time_pct = {"positive": 0.0, "neutral": 100.0, "negative": 0.0}
        global_sent_probs = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    dom_sent = max(sent_time_pct, key=sent_time_pct.get)
    data["dominant_sentiment"] = dom_sent
    data["dominant_sentiment_probability"] = global_sent_probs.get(dom_sent, 0.5)
    data["sentiment_time_distribution_percent"] = sent_time_pct
    data["sentiment_counts"] = sent_counts
    data["global_sentiment_probability_vector"] = global_sent_probs

    fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    # 1. Backfill Ground Truth files
    gt2000_path = Path("data/audio_samples/ground_truth_2000_cleaned.json")
    gt2000 = json.loads(gt2000_path.read_text(encoding="utf-8"))
    gt_map = {}
    for k_list in ["english_samples", "japanese_samples"]:
        for s in gt2000.get(k_list, []):
            s["ground_truth_sentiment"] = emo_to_sentiment(s["ground_truth_emotion"])
            gt_map[s["sample_id"]] = s
    gt2000_path.write_text(json.dumps(gt2000, indent=2, ensure_ascii=False), encoding="utf-8")

    gt60_path = Path("data/audio_samples/ground_truth_60_cleaned.json")
    gt60_map = {}
    if gt60_path.exists():
        gt60 = json.loads(gt60_path.read_text(encoding="utf-8"))
        for k_list in ["english_samples", "japanese_samples"]:
            for s in gt60.get(k_list, []):
                s["ground_truth_sentiment"] = emo_to_sentiment(s["ground_truth_emotion"])
                gt60_map[s["sample_id"]] = s
        gt60_path.write_text(json.dumps(gt60, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. Backfill 2,000-clip per-model record files
    for fname in ["benchmark_kev_0.8b_2000_results.json", "benchmark_kev_4b_2000_results.json"]:
        fpath = Path("benchmarks/results") / fname
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            recs = data.get("records", [])
            en_s_ok = ja_s_ok = 0
            en_tot = ja_tot = 0
            for r in recs:
                exp_s = emo_to_sentiment(r["expected_emotion"])
                pred_s = emo_to_sentiment(r["predicted_emotion"], bool(r.get("sarcasm_flag", False)))
                s_match = exp_s == pred_s
                r["expected_sentiment"] = exp_s
                r["predicted_sentiment"] = pred_s
                r["sentiment_match"] = s_match
                if r.get("language") == "en":
                    en_tot += 1
                    if s_match:
                        en_s_ok += 1
                else:
                    ja_tot += 1
                    if s_match:
                        ja_s_ok += 1
            tot = max(1, en_tot + ja_tot)
            data["overall_sentiment_accuracy"] = round((en_s_ok + ja_s_ok) / tot, 4)
            data["english_sentiment_accuracy"] = round(en_s_ok / max(1, en_tot), 4)
            data["japanese_sentiment_accuracy"] = round(ja_s_ok / max(1, ja_tot), 4)
            data["overall_sentiment_correct"] = en_s_ok + ja_s_ok
            data["english_sentiment_correct"] = en_s_ok
            data["japanese_sentiment_correct"] = ja_s_ok
            fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"[2000] {fname}: Sentiment Overall={data['overall_sentiment_accuracy']*100:.2f}% "
                f"(EN={data['english_sentiment_accuracy']*100:.2f}%, JA={data['japanese_sentiment_accuracy']*100:.2f}%)"
            )

    # 3. Compute E2V+ standalone 2000 sentiment accuracy
    ac_cache = json.loads(Path("data/audio_samples/acoustics_2000_cache.json").read_text(encoding="utf-8"))
    ser_cache = ac_cache.get("ser", {})
    E2V_MAP = {
        "neutral": "neutral_inquiry",
        "happy": "satisfaction_gratitude",
        "angry": "outrage_hostility",
        "sad": "disappointed_sad",
        "fearful": "urgency_panic",
        "disgusted": "frustration_irritation",
        "surprised": "anxiety_confusion",
        "<unk>": "neutral_inquiry",
        "unk": "neutral_inquiry",
        "other": "neutral_inquiry",
    }
    e2v_en_s = e2v_ja_s = 0
    e2v_60_en_s = e2v_60_ja_s = 0
    for sid, s in gt_map.items():
        lbl = ser_cache.get(sid, {}).get("top_label", "neutral")
        if "/" in lbl:
            lbl = lbl.split("/")[-1].strip().lower()
        e2v_emo = E2V_MAP.get(lbl, "neutral_inquiry")
        s_match = emo_to_sentiment(e2v_emo) == s["ground_truth_sentiment"]
        if s_match:
            if s["language"] == "en":
                e2v_en_s += 1
            else:
                e2v_ja_s += 1
        if sid in gt60_map and s_match:
            if s["language"] == "en":
                e2v_60_en_s += 1
            else:
                e2v_60_ja_s += 1

    # 4. Update benchmark_2000_summary_results.json
    sum2000_path = Path("benchmarks/results/benchmark_2000_summary_results.json")
    sum2000 = json.loads(sum2000_path.read_text(encoding="utf-8"))
    k08 = json.loads(Path("benchmarks/results/benchmark_kev_0.8b_2000_results.json").read_text(encoding="utf-8"))
    k4b = json.loads(Path("benchmarks/results/benchmark_kev_4b_2000_results.json").read_text(encoding="utf-8"))

    for m in sum2000.get("models", []):
        name = m.get("model", "")
        if "Emotion2Vec+" in name:
            m["english_sentiment_accuracy"] = round(e2v_en_s / 1000, 4)
            m["japanese_sentiment_accuracy"] = round(e2v_ja_s / 1000, 4)
            m["overall_sentiment_accuracy"] = round((e2v_en_s + e2v_ja_s) / 2000, 4)
            m["english_sentiment_correct"] = e2v_en_s
            m["japanese_sentiment_correct"] = e2v_ja_s
            m["overall_sentiment_correct"] = e2v_en_s + e2v_ja_s
        elif "0.8B" in name and "Best" not in name:
            m["english_sentiment_accuracy"] = k08["english_sentiment_accuracy"]
            m["japanese_sentiment_accuracy"] = k08["japanese_sentiment_accuracy"]
            m["overall_sentiment_accuracy"] = k08["overall_sentiment_accuracy"]
            m["english_sentiment_correct"] = k08["english_sentiment_correct"]
            m["japanese_sentiment_correct"] = k08["japanese_sentiment_correct"]
            m["overall_sentiment_correct"] = k08["overall_sentiment_correct"]
        elif "4B" in name and "Best" not in name:
            m["english_sentiment_accuracy"] = k4b["english_sentiment_accuracy"]
            m["japanese_sentiment_accuracy"] = k4b["japanese_sentiment_accuracy"]
            m["overall_sentiment_accuracy"] = k4b["overall_sentiment_accuracy"]
            m["english_sentiment_correct"] = k4b["english_sentiment_correct"]
            m["japanese_sentiment_correct"] = k4b["japanese_sentiment_correct"]
            m["overall_sentiment_correct"] = k4b["overall_sentiment_correct"]
        elif "Best" in name:
            best_en = max(k08["english_sentiment_correct"], k4b["english_sentiment_correct"])
            best_ja = max(k08["japanese_sentiment_correct"], k4b["japanese_sentiment_correct"])
            m["english_sentiment_accuracy"] = round(best_en / 1000, 4)
            m["japanese_sentiment_accuracy"] = round(best_ja / 1000, 4)
            m["overall_sentiment_accuracy"] = round((best_en + best_ja) / 2000, 4)
            m["english_sentiment_correct"] = best_en
            m["japanese_sentiment_correct"] = best_ja
            m["overall_sentiment_correct"] = best_en + best_ja
        print(
            f"[Summary 2000] {name}: 8-Emo={m['overall_accuracy']*100:.2f}% -> "
            f"3-Sentiment={m['overall_sentiment_accuracy']*100:.2f}% "
            f"(EN={m['english_sentiment_accuracy']*100:.2f}%, JA={m['japanese_sentiment_accuracy']*100:.2f}%)"
        )
    sum2000_path.write_text(json.dumps(sum2000, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Backfill 60-clip benchmark files
    map_60_files = {
        "Laya mmBERT": "benchmark_60_audio_results.json",
        "Kev-0.8B FP16": "benchmark_kev_0.8b_results.json",
        "Kev-0.8B INT4 (NF4)": "benchmark_kev_0.8b_int4_results.json",
        "Kev-4B INT4 (Q4_K_M)": "benchmark_kev_4b_results.json",
    }
    stats_60 = {
        "Emotion2Vec+ Large Standalone": (e2v_60_en_s, e2v_60_ja_s),
    }
    for m_label, fname in map_60_files.items():
        fpath = Path("benchmarks/results") / fname
        if fpath.exists():
            d60 = json.loads(fpath.read_text(encoding="utf-8"))
            en_ok = ja_ok = 0
            for r in d60.get("records", []):
                sid = r.get("sample_id")
                exp_emo = gt60_map.get(sid, {}).get("ground_truth_emotion", r.get("expected_emotion", "neutral_inquiry"))
                exp_s = emo_to_sentiment(exp_emo)
                pred_s = emo_to_sentiment(r.get("predicted_emotion", "neutral_inquiry"), bool(r.get("sarcasm_flag", False)))
                s_match = exp_s == pred_s
                r["expected_sentiment"] = exp_s
                r["predicted_sentiment"] = pred_s
                r["sentiment_match"] = s_match
                lang = r.get("language") or ("en" if sid.startswith("en") else "ja")
                if lang == "en" and s_match:
                    en_ok += 1
                elif lang == "ja" and s_match:
                    ja_ok += 1
            d60["en_sentiment_acc"] = round(en_ok / 30, 4)
            d60["ja_sentiment_acc"] = round(ja_ok / 30, 4)
            d60["overall_sentiment_acc"] = round((en_ok + ja_ok) / 60, 4)
            fpath.write_text(json.dumps(d60, indent=2, ensure_ascii=False), encoding="utf-8")
            stats_60[m_label] = (en_ok, ja_ok)

    rerun60_path = Path("benchmarks/results/benchmark_rerun_cleaned_results.json")
    if rerun60_path.exists():
        r60 = json.loads(rerun60_path.read_text(encoding="utf-8"))
        for item in r60.get("results", []):
            mname = item.get("model", "")
            if mname in stats_60:
                en_ok, ja_ok = stats_60[mname]
                item["en_sentiment_acc"] = round(en_ok / 30, 4)
                item["en_sentiment_correct"] = en_ok
                item["ja_sentiment_acc"] = round(ja_ok / 30, 4)
                item["ja_sentiment_correct"] = ja_ok
                item["overall_sentiment_acc"] = round((en_ok + ja_ok) / 60, 4)
                item["overall_sentiment_correct"] = en_ok + ja_ok
                print(
                    f"[60 Audited] {mname}: 8-Emo={item['overall_acc']*100:.1f}% -> "
                    f"3-Sentiment={item['overall_sentiment_acc']*100:.1f}% "
                    f"(EN={item['en_sentiment_acc']*100:.1f}%, JA={item['ja_sentiment_acc']*100:.1f}%)"
                )
        rerun60_path.write_text(json.dumps(r60, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. Backfill combined call files & saved_recordings
    for pfile in [
        Path("benchmarks/results/combined_english_pipeline_results.json"),
        Path("benchmarks/results/combined_japanese_pipeline_results.json"),
        Path("combined_english_pipeline_results.json"),
        Path("combined_japanese_pipeline_results.json"),
    ]:
        backfill_pipeline_json(pfile)

    saved_dir = Path("data/saved_recordings")
    if saved_dir.exists():
        for rec_f in saved_dir.glob("rec_*.json"):
            backfill_pipeline_json(rec_f)
        hist_f = saved_dir / "history.json"
        if hist_f.exists():
            hist = json.loads(hist_f.read_text(encoding="utf-8"))
            for h in hist:
                rec_json = saved_dir / f"{h['id']}.json"
                if rec_json.exists():
                    rj = json.loads(rec_json.read_text(encoding="utf-8"))
                    h["dominant_emotion"] = rj.get("dominant_emotion", h.get("dominant_emotion"))
                    h["dominant_sentiment"] = rj.get("dominant_sentiment", emo_to_sentiment(h["dominant_emotion"]))
            hist_f.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[✓] All benchmarks and saved recordings backfilled with 3-way sentiment!")


if __name__ == "__main__":
    main()
