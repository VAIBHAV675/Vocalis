import json

def eval_model(records, gt_dict, model_name, pred_key="predicted_emotion"):
    en_total = 0
    en_correct = 0
    ja_total = 0
    ja_correct = 0

    for r in records:
        sid = r["sample_id"]
        if sid not in gt_dict:
            continue
        expected = gt_dict[sid]
        pred = r[pred_key]
        match = (pred == expected)
        lang = r["language"]
        if lang == "en":
            en_total += 1
            if match:
                en_correct += 1
        elif lang == "ja":
            ja_total += 1
            if match:
                ja_correct += 1

    total = en_total + ja_total
    correct = en_correct + ja_correct
    return {
        "model": model_name,
        "en_acc": en_correct / en_total if en_total else 0,
        "en_correct": en_correct,
        "en_total": en_total,
        "ja_acc": ja_correct / ja_total if ja_total else 0,
        "ja_correct": ja_correct,
        "ja_total": ja_total,
        "overall_acc": correct / total if total else 0,
        "overall_correct": correct,
        "overall_total": total,
    }

def main():
    with open("data/audio_samples/ground_truth_60_cleaned.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_dict = {s["sample_id"]: s["ground_truth_emotion"] for s in (gt_data["english_samples"] + gt_data["japanese_samples"])}

    # 1. Emotion2Vec+ Large Standalone
    with open("benchmark_60_audio_results.json", "r", encoding="utf-8") as f:
        laya_data = json.load(f)
    e2v_map = {'neutral': 'neutral_inquiry', 'happy': 'satisfaction_gratitude', 'angry': 'outrage_hostility', 'sad': 'disappointed_sad', 'fearful': 'urgency_panic'}
    e2v_records = []
    for r in laya_data["records"]:
        e2v_records.append({
            "sample_id": r["sample_id"],
            "language": r["language"],
            "pred": e2v_map.get(r["ser_prior"], "neutral_inquiry")
        })
    res_e2v = eval_model(e2v_records, gt_dict, "Emotion2Vec+ Large Standalone", pred_key="pred")

    # 2. Laya mmBERT
    res_laya = eval_model(laya_data["records"], gt_dict, "Laya mmBERT")

    # 3. Kev-0.8B FP16
    with open("benchmark_kev_0.8b_results.json", "r", encoding="utf-8") as f:
        k08_fp16 = json.load(f)
    res_k08_fp16 = eval_model(k08_fp16["records"], gt_dict, "Kev-0.8B FP16")

    # 4. Kev-0.8B INT4
    with open("benchmark_kev_0.8b_int4_results.json", "r", encoding="utf-8") as f:
        k08_int4 = json.load(f)
    res_k08_int4 = eval_model(k08_int4["records"], gt_dict, "Kev-0.8B INT4 (NF4)")

    # 5. Kev-4B INT4
    with open("benchmark_kev_4b_results.json", "r", encoding="utf-8") as f:
        k4_int4 = json.load(f)
    res_k4_int4 = eval_model(k4_int4["records"], gt_dict, "Kev-4B INT4 (Q4_K_M)")

    models = [res_e2v, res_laya, res_k08_fp16, res_k08_int4, res_k4_int4]

    print("=" * 86)
    print("RERUN EVALUATION RESULTS ACROSS AUDITED & CLEANED GROUND TRUTH (60 SAMPLES)")
    print("=" * 86)
    print(f"{'Model Name':<32} | {'Overall Accuracy':<18} | {'English (30)':<14} | {'Japanese (30)':<14}")
    print("-" * 86)
    for m in models:
        print(f"{m['model']:<32} | {m['overall_acc']:>6.1%} ({m['overall_correct']}/{m['overall_total']})     | {m['en_acc']:>6.1%} ({m['en_correct']}/{m['en_total']}) | {m['ja_acc']:>6.1%} ({m['ja_correct']}/{m['ja_total']})")
    print("=" * 86)

    output = {
        "benchmark": "Cleaned Multilingual Ground Truth (60 samples)",
        "results": models
    }
    with open("benchmark_rerun_cleaned_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\n[+] Saved rerun results to benchmark_rerun_cleaned_results.json")

if __name__ == "__main__":
    main()
