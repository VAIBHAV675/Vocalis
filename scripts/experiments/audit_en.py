import json

with open("data/audio_samples/ground_truth_60_multilingual.json", encoding="utf-8") as f:
    gt = json.load(f)
with open("benchmark_kev_4b_results.json", encoding="utf-8") as f:
    k4 = {r["sample_id"]: r for r in json.load(f)["records"]}

print("=== ALL 30 ENGLISH SAMPLES AUDIT ===")
for s in gt["english_samples"]:
    sid = s["sample_id"]
    r = k4[sid]
    status = "CORRECT" if r["match"] else "MISMATCH"
    print(f"[{sid}] Expected: {s['ground_truth_emotion']} | Kev-4B: {r['predicted_emotion']} | Match: {status}")
    print(f"       Text: \"{s['ground_truth_transcript']}\"")
