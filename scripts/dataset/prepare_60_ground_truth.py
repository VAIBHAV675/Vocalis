import json
from pathlib import Path

# Ground truth dataset mapping for all 60 audio samples
# 30 English (Earnings22) + 30 Japanese (ReazonSpeech)
# Mapped against our canonical 8 emotional categories:
# 1. satisfaction_gratitude
# 2. neutral_inquiry
# 3. frustration_irritation
# 4. outrage_hostility
# 5. disappointed_sad
# 6. sarcasm_passive_aggression
# 7. anxiety_confusion
# 8. urgency_panic

EN_GROUND_TRUTH = {
    "en_ib_sample_01": "neutral_inquiry",
    "en_ib_sample_02": "neutral_inquiry",
    "en_ib_sample_03": "neutral_inquiry",
    "en_ib_sample_04": "satisfaction_gratitude",
    "en_ib_sample_05": "neutral_inquiry",
    "en_ib_sample_06": "satisfaction_gratitude",
    "en_ib_sample_07": "satisfaction_gratitude",
    "en_ib_sample_08": "neutral_inquiry",
    "en_ib_sample_09": "satisfaction_gratitude",
    "en_ib_sample_10": "neutral_inquiry",
    "en_ib_sample_11": "satisfaction_gratitude",
    "en_ib_sample_12": "satisfaction_gratitude",
    "en_ib_sample_13": "neutral_inquiry",
    "en_ib_sample_14": "neutral_inquiry",
    "en_ib_sample_15": "disappointed_sad",
    "en_ib_sample_16": "satisfaction_gratitude",
    "en_ib_sample_17": "neutral_inquiry",
    "en_ib_sample_18": "satisfaction_gratitude",
    "en_ib_sample_19": "satisfaction_gratitude",
    "en_ib_sample_20": "neutral_inquiry",
    "en_ib_sample_21": "neutral_inquiry",
    "en_ib_sample_22": "neutral_inquiry",
    "en_ib_sample_23": "neutral_inquiry",
    "en_ib_sample_24": "satisfaction_gratitude",
    "en_ib_sample_25": "neutral_inquiry",
    "en_ib_sample_26": "neutral_inquiry",
    "en_ib_sample_27": "frustration_irritation",
    "en_ib_sample_28": "neutral_inquiry",
    "en_ib_sample_29": "neutral_inquiry",
    "en_ib_sample_30": "neutral_inquiry"
}

JA_GROUND_TRUTH = {
    "ja_ib_sample_01": "neutral_inquiry",
    "ja_ib_sample_02": "neutral_inquiry",
    "ja_ib_sample_03": "anxiety_confusion",
    "ja_ib_sample_04": "neutral_inquiry",
    "ja_ib_sample_05": "frustration_irritation",
    "ja_ib_sample_06": "neutral_inquiry",
    "ja_ib_sample_07": "urgency_panic",
    "ja_ib_sample_08": "neutral_inquiry",
    "ja_ib_sample_09": "frustration_irritation",
    "ja_ib_sample_10": "satisfaction_gratitude",
    "ja_ib_sample_11": "neutral_inquiry",
    "ja_ib_sample_12": "anxiety_confusion",
    "ja_ib_sample_13": "neutral_inquiry",
    "ja_ib_sample_14": "satisfaction_gratitude",
    "ja_ib_sample_15": "disappointed_sad",
    "ja_ib_sample_16": "frustration_irritation",
    "ja_ib_sample_17": "satisfaction_gratitude",
    "ja_ib_sample_18": "disappointed_sad",
    "ja_ib_sample_19": "urgency_panic",
    "ja_ib_sample_20": "satisfaction_gratitude",
    "ja_ib_sample_21": "neutral_inquiry",
    "ja_ib_sample_22": "neutral_inquiry",
    "ja_ib_sample_23": "anxiety_confusion",
    "ja_ib_sample_24": "satisfaction_gratitude",
    "ja_ib_sample_25": "satisfaction_gratitude",
    "ja_ib_sample_26": "satisfaction_gratitude",
    "ja_ib_sample_27": "neutral_inquiry",
    "ja_ib_sample_28": "neutral_inquiry",
    "ja_ib_sample_29": "disappointed_sad",
    "ja_ib_sample_30": "neutral_inquiry"
}

# Load existing JSON files to combine with emotional labels
en_samples = json.load(open("data/audio_samples/en/ground_truth_en.json", "r", encoding="utf-8"))
for s in en_samples:
    sid = s["sample_id"]
    s["ground_truth_emotion"] = EN_GROUND_TRUTH.get(sid, "neutral_inquiry")
    s["wav_path"] = f"data/audio_samples/en/{s['filename']}"

ja_samples = json.load(open("data/audio_samples/ja/ground_truth_ja.json", "r", encoding="utf-8"))
for s in ja_samples:
    sid = s["sample_id"]
    s["ground_truth_emotion"] = JA_GROUND_TRUTH.get(sid, "neutral_inquiry")
    s["wav_path"] = f"data/audio_samples/ja/{s['filename']}"

full_dataset = {
    "total_samples": len(en_samples) + len(ja_samples),
    "english_samples": en_samples,
    "japanese_samples": ja_samples
}

with open("data/audio_samples/ground_truth_60_multilingual.json", "w", encoding="utf-8") as f:
    json.dump(full_dataset, f, indent=2, ensure_ascii=False)

print(f"Successfully generated ground truth dataset with {full_dataset['total_samples']} samples!")
