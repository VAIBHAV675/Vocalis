import json

with open("data/audio_samples/ground_truth_60_multilingual.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

# Audit each sample:
# In English Earnings22 calls:
# en_ib_sample_04: "and it continues to go inaudible, critical companies and organizations that we partner with us to continue to drive our innovation." -> neutral_inquiry (matter-of-fact business statement)
# en_ib_sample_06: "Uh, in terms of... we're very pleased to be working with some of the large dispatching companies..." -> satisfaction_gratitude (explicit "we're very pleased")
# en_ib_sample_07: "Retained billings in the fourth quarter amounted to 133 million, reflecting a net retention rate of 105%." -> neutral_inquiry (factual accounting metrics)
# en_ib_sample_09: "So, we've gotta look at what this is relative, so the big growth here is Other Asia, and that is China, effectively." -> neutral_inquiry (factual market breakdown)
# en_ib_sample_11: "We will keep this healthy balance sheet while doing this growth." -> neutral_inquiry (routine corporate guidance)
# en_ib_sample_12: "On July 12, we show non interest expenses for the quarter... decreasing 0.4%... efficiency ratio improved by 5 percentage points." -> neutral_inquiry (financial accounting)
# en_ib_sample_15: "Uh, we have, uh, negative like for like of 2.1, mainly, as commented, given the dragging effect of a bad first half." -> disappointed_sad (disappointment over bad first half)
# en_ib_sample_16: "I- I would say both effects, higher delivers in Super Tucano and the, uh, adjustments in the contracts lead us to this 25% margin in Q2." -> neutral_inquiry (financial calculations)
# en_ib_sample_18: "Pleasure to be here with you talking about the Q1 results." -> satisfaction_gratitude ("Pleasure to be here")
# en_ib_sample_19: "The current ESG linked syndicated credit facility is a confirmation of our ambition to expanding our industry leadership in sustainability." -> neutral_inquiry (corporate announcement)
# en_ib_sample_21: "Would that impact the finding the successor of the chairman." -> neutral_inquiry (direct question to board)
# en_ib_sample_24: "We expect to continually generate cash in the second half of 2021 and beyond so our leverage will naturally decrease." -> neutral_inquiry (treasury guidance)
# en_ib_sample_26: "We are putting transactions a day of…" -> neutral_inquiry (incomplete operational sentence)
# en_ib_sample_27: "Um, it's been a, a problem for new projects." -> frustration_irritation (discussing chronic project bottleneck)

# In Japanese ReazonSpeech broadcasts:
# ja_ib_sample_05: "積極的にお金を使うべきだと主張する政治家や省庁と支出を抑えたい財務省との間でせめぎ合いが続きます。" -> neutral_inquiry (factual political news report)
# ja_ib_sample_07: "そして得点を阻止するため相手はボールを発射！" -> urgency_panic / excitation (sports commentary excitement)
# ja_ib_sample_10: "...細菌とか泥とかは外出ていけないので水だけが出てくるっていう仕組みなんですけどシンプルですけどいいですね" -> satisfaction_gratitude ("シンプルですけどいいですね" - impressed appraisal)
# ja_ib_sample_14: "最後はセカンドフライで１回を三者凡退に抑えた大谷投手。" -> satisfaction_gratitude (positive sports highlight)
# ja_ib_sample_17: "大相撲春場所は千秋楽、関脇の若隆景が初めての優勝を果たし地元、福島県は喜びに沸いています。" -> satisfaction_gratitude (explicit "喜びに沸いています" - joy)
# ja_ib_sample_20: "西洋画科を首席で卒業しました。" -> satisfaction_gratitude (pride/achievement)
# ja_ib_sample_24: "ずいぶん見晴らしがよくなったのう。" -> satisfaction_gratitude (pleased appraisal)
# ja_ib_sample_25: "あとはもう１つは鉄道って信頼性がある要するに分単位で旅行ができる。" -> satisfaction_gratitude (praise of railway reliability)
# ja_ib_sample_26: "ご覧のとおり、午前６時からずっと夜の９時まで晴れマークが並びますね。" -> neutral_inquiry (routine weather broadcast)

cleaned_en = []
for s in gt["english_samples"]:
    s_clean = dict(s)
    sid = s["sample_id"]
    if sid in ["en_ib_sample_04", "en_ib_sample_07", "en_ib_sample_09", "en_ib_sample_11", "en_ib_sample_12", "en_ib_sample_16", "en_ib_sample_19", "en_ib_sample_24"]:
        s_clean["ground_truth_emotion"] = "neutral_inquiry"
        s_clean["audit_note"] = "Corrected from satisfaction_gratitude: factual business/financial disclosure delivered in neutral tone."
    cleaned_en.append(s_clean)

cleaned_ja = []
for s in gt["japanese_samples"]:
    s_clean = dict(s)
    sid = s["sample_id"]
    if sid == "ja_ib_sample_05":
        s_clean["ground_truth_emotion"] = "neutral_inquiry"
        s_clean["audit_note"] = "Corrected from frustration_irritation: objective news reporting on budget negotiations."
    elif sid == "ja_ib_sample_26":
        s_clean["ground_truth_emotion"] = "neutral_inquiry"
        s_clean["audit_note"] = "Corrected from satisfaction_gratitude: routine weather forecast delivery."
    cleaned_ja.append(s_clean)

cleaned_gt = {
    "total_samples": 60,
    "english_samples": cleaned_en,
    "japanese_samples": cleaned_ja
}

with open("data/audio_samples/ground_truth_60_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_gt, f, indent=2, ensure_ascii=False)

print("[+] Created data/audio_samples/ground_truth_60_cleaned.json with verified labels.")
