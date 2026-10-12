import json

with open("data/audio_samples/ground_truth_60_multilingual.json", encoding="utf-8") as f:
    gt = json.load(f)
with open("benchmark_kev_0.8b_results.json", encoding="utf-8") as f:
    k08 = {r["sample_id"]: r["predicted_emotion"] for r in json.load(f)["records"]}
with open("benchmark_kev_4b_results.json", encoding="utf-8") as f:
    k4 = {r["sample_id"]: r["predicted_emotion"] for r in json.load(f)["records"]}

print("=== ALL SAMPLES LABELED satisfaction_gratitude IN ENGLISH ===")
for s in gt["english_samples"]:
    if s["ground_truth_emotion"] == "satisfaction_gratitude":
        sid = s["sample_id"]
        print(f"[{sid}]")
        print(f"  Transcript:   \"{s['ground_truth_transcript']}\"")
        print(f"  Ground Truth: {s['ground_truth_emotion']}")
        print(f"  Kev-0.8B:     {k08.get(sid)}")
        print(f"  Kev-4B:       {k4.get(sid)}")
        print()
