import os
import json
import time
from pathlib import Path

# Load .env & configuration
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from laya import Router

# Initialize and preload
print("=" * 70)
print("PRELOADING LAYA MULTILINGUAL CHECKPOINT")
print("=" * 70)
t_start = time.time()
router = Router(preload=False)
router.preload(["multilingual"])
print(f"Preload finished in {time.time() - t_start:.2f}s\n")

# 1. FIXED, CONSTANT PROMPT / QUESTIONS (Standardized across all requests)
CONSTANT_QUESTIONS = {
    "final_sentiment": {
        "type": "choice",
        "instructions": (
            "Analyze the Japanese customer interaction and determine true emotional sentiment. "
            "Resolve any discrepancy between spoken text and acoustic vocal tone. "
            "If acoustic cues indicate anger, hostility, or sarcasm despite polite words, prioritize acoustic tone."
        ),
        "criteria": {
            "positive": "Genuine satisfaction, gratitude, happiness, or agreement.",
            "neutral": "Routine business dialogue, calm inquiry, matter-of-fact tone.",
            "frustrated_hostile": "Irritation, suppressed anger, sarcasm, passive-aggression, or open hostility.",
            "disappointed_sad": "Sorrow, disappointment, regret, helplessness, or sadness."
        }
    },
    "is_sarcastic_or_conflicted": {
        "type": "noul",
        "instructions": (
            "Is there a contradiction between polite/neutral words and negative acoustic voice tone (sarcasm or masked aggression)?"
        )
    }
}

# 2. STANDARDIZED TEMPLATE FUNCTION (Fixed structure)
def format_input(transcript: str, acoustic_emotion: str, acoustic_conf: float, prosody: str) -> str:
    return (
        f"Speaker text: 「{transcript}」\n"
        f"Acoustic vocal tone: {acoustic_emotion} (confidence: {acoustic_conf:.0%})\n"
        f"Prosody cues: {prosody}"
    )

# 3. 26 REALISTIC TEST BENCHMARK CASES
test_cases = [
    # Category 1: Sarcasm / Masked Frustration (Polite words + Angry acoustics) -> Expected: frustrated_hostile, Conflict: True
    {
        "id": "SARC_01",
        "category": "sarcasm",
        "transcript": "ご対応いただき、誠にありがとうございました。",
        "acoustic_emotion": "anger",
        "acoustic_conf": 0.85,
        "prosody": "high vocal tension, sharp pitch attack, compressed tone",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },
    {
        "id": "SARC_02",
        "category": "sarcasm",
        "transcript": "はいはい、よく分かりました。",
        "acoustic_emotion": "anger / irritation",
        "acoustic_conf": 0.82,
        "prosody": "exasperated sigh, sharp downward inflection, rapid cadence",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },
    {
        "id": "SARC_03",
        "category": "sarcasm",
        "transcript": "大変素晴らしいサービスですね。",
        "acoustic_emotion": "cold sarcasm / hostility",
        "acoustic_conf": 0.88,
        "prosody": "cold monotone, flat pitch attack, strained vocal cords",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },
    {
        "id": "SARC_04",
        "category": "sarcasm",
        "transcript": "お忙しいところ恐れ入ります。",
        "acoustic_emotion": "irritation",
        "acoustic_conf": 0.79,
        "prosody": "high vocal tension, aggressive pitch spike, harsh breathing",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },
    {
        "id": "SARC_05",
        "category": "sarcasm",
        "transcript": "結構です、もう十分伝わりましたので。",
        "acoustic_emotion": "anger",
        "acoustic_conf": 0.90,
        "prosody": "low aggressive growl, abrupt cutoff, heavy breath",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },
    {
        "id": "SARC_06",
        "category": "sarcasm",
        "transcript": "どうぞごゆっくりご検討くださいませ。",
        "acoustic_emotion": "passive-aggression",
        "acoustic_conf": 0.84,
        "prosody": "sarcastic drawl, heavy cynical sigh, forced cadence",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": True
    },

    # Category 2: Genuine Positive (Polite words + Joy/Calm acoustics) -> Expected: positive, Conflict: False
    {
        "id": "POS_01",
        "category": "positive",
        "transcript": "本当に助かりました！ありがとうございます。",
        "acoustic_emotion": "joy / gratitude",
        "acoustic_conf": 0.92,
        "prosody": "high pitch variation, warm laughter, bright acoustic energy",
        "expected_sentiment": "positive",
        "expected_conflict": False
    },
    {
        "id": "POS_02",
        "category": "positive",
        "transcript": "迅速にご対応いただき感謝申し上げます。",
        "acoustic_emotion": "joy",
        "acoustic_conf": 0.86,
        "prosody": "warm relaxed cadence, bright vocal energy, smiling resonance",
        "expected_sentiment": "positive",
        "expected_conflict": False
    },
    {
        "id": "POS_03",
        "category": "positive",
        "transcript": "わかりやすく説明していただき助かりました。",
        "acoustic_emotion": "satisfaction",
        "acoustic_conf": 0.89,
        "prosody": "calm melodic cadence, smooth articulation, relaxed breathing",
        "expected_sentiment": "positive",
        "expected_conflict": False
    },
    {
        "id": "POS_04",
        "category": "positive",
        "transcript": "素晴らしいご提案をいただき嬉しいです。",
        "acoustic_emotion": "joy",
        "acoustic_conf": 0.94,
        "prosody": "upbeat intonation, smiling voice, elevated pitch contour",
        "expected_sentiment": "positive",
        "expected_conflict": False
    },
    {
        "id": "POS_05",
        "category": "positive",
        "transcript": "おかげさまで無事に問題が解決いたしました！",
        "acoustic_emotion": "relief / joy",
        "acoustic_conf": 0.91,
        "prosody": "lively energy, bright pitch lift, enthusiastic tempo",
        "expected_sentiment": "positive",
        "expected_conflict": False
    },

    # Category 3: Neutral Business Inquiries (Standard words + Neutral acoustics) -> Expected: neutral, Conflict: False
    {
        "id": "NEUT_01",
        "category": "neutral",
        "transcript": "来週の営業時間は何時からでしょうか。",
        "acoustic_emotion": "neutral",
        "acoustic_conf": 0.95,
        "prosody": "steady pitch, normal conversational tempo, balanced energy",
        "expected_sentiment": "neutral",
        "expected_conflict": False
    },
    {
        "id": "NEUT_02",
        "category": "neutral",
        "transcript": "請求書の送付先住所を変更したいのですが。",
        "acoustic_emotion": "neutral",
        "acoustic_conf": 0.92,
        "prosody": "calm matter-of-fact tone, standard business intonation",
        "expected_sentiment": "neutral",
        "expected_conflict": False
    },
    {
        "id": "NEUT_03",
        "category": "neutral",
        "transcript": "現在の契約プランの詳細について確認させてください。",
        "acoustic_emotion": "neutral",
        "acoustic_conf": 0.94,
        "prosody": "even pace, flat baseline pitch, neutral cadence",
        "expected_sentiment": "neutral",
        "expected_conflict": False
    },
    {
        "id": "NEUT_04",
        "category": "neutral",
        "transcript": "パスワードの再設定手順を教えていただけますでしょうか。",
        "acoustic_emotion": "neutral",
        "acoustic_conf": 0.90,
        "prosody": "polite formal cadence, relaxed pitch, steady breathing",
        "expected_sentiment": "neutral",
        "expected_conflict": False
    },
    {
        "id": "NEUT_05",
        "category": "neutral",
        "transcript": "資料をPDF形式でメール送付いただくことは可能ですか。",
        "acoustic_emotion": "neutral",
        "acoustic_conf": 0.93,
        "prosody": "calm conversational, regular pacing, clear enunciation",
        "expected_sentiment": "neutral",
        "expected_conflict": False
    },

    # Category 4: Open Hostility / Outrage (Angry words + Angry acoustics) -> Expected: frustrated_hostile, Conflict: False
    {
        "id": "HOST_01",
        "category": "hostile",
        "transcript": "ふざけるな！何度同じことを言わせるんだ！",
        "acoustic_emotion": "furious anger",
        "acoustic_conf": 0.98,
        "prosody": "loud shouting, explosive attack, peak vocal energy, heavy tremor",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": False
    },
    {
        "id": "HOST_02",
        "category": "hostile",
        "transcript": "責任者を今すぐ出してください！話になりません。",
        "acoustic_emotion": "anger",
        "acoustic_conf": 0.95,
        "prosody": "harsh sharp pitch, shaking tense voice, rapid aggressive cadence",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": False
    },
    {
        "id": "HOST_03",
        "category": "hostile",
        "transcript": "全く使い物にならない。時間の無駄だった。",
        "acoustic_emotion": "irritation / disgust",
        "acoustic_conf": 0.91,
        "prosody": "snapping tone, vocal fry, deep irritation, abrupt breath",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": False
    },
    {
        "id": "HOST_04",
        "category": "hostile",
        "transcript": "不良品を売りつけておいてその対応は何だ！返金しろ！",
        "acoustic_emotion": "anger",
        "acoustic_conf": 0.97,
        "prosody": "raised volume, clipping acoustic energy, harsh attack",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": False
    },
    {
        "id": "HOST_05",
        "category": "hostile",
        "transcript": "約束期日を過ぎているのに連絡もないとはどういう了見だ！",
        "acoustic_emotion": "anger",
        "acoustic_conf": 0.94,
        "prosody": "strained aggressive projection, elevated pitch peak, rapid speech",
        "expected_sentiment": "frustrated_hostile",
        "expected_conflict": False
    },

    # Category 5: Disappointment / Sadness / Resignation (Sad words + Sad acoustics) -> Expected: disappointed_sad, Conflict: False
    {
        "id": "SAD_01",
        "category": "disappointed",
        "transcript": "そうですか…期待していたのですが本当に残念です。",
        "acoustic_emotion": "sadness / disappointment",
        "acoustic_conf": 0.88,
        "prosody": "low vocal energy, falling pitch contour, trailing heavy sigh",
        "expected_sentiment": "disappointed_sad",
        "expected_conflict": False
    },
    {
        "id": "SAD_02",
        "category": "disappointed",
        "transcript": "もうデータの復旧はできないのですね…困りました。",
        "acoustic_emotion": "distress / sadness",
        "acoustic_conf": 0.86,
        "prosody": "hesitant slow tempo, quivering pitch, suppressed weeping breath",
        "expected_sentiment": "disappointed_sad",
        "expected_conflict": False
    },
    {
        "id": "SAD_03",
        "category": "disappointed",
        "transcript": "仕方ないですね…今回は諦めることにします。",
        "acoustic_emotion": "resignation / sadness",
        "acoustic_conf": 0.91,
        "prosody": "subdued volume, flat downtoned cadence, slow articulation",
        "expected_sentiment": "disappointed_sad",
        "expected_conflict": False
    },
    {
        "id": "SAD_04",
        "category": "disappointed",
        "transcript": "楽しみに待っていただけにショックが大きいです…",
        "acoustic_emotion": "sadness",
        "acoustic_conf": 0.89,
        "prosody": "muted tone, weak vocal resonance, low pitch tremble",
        "expected_sentiment": "disappointed_sad",
        "expected_conflict": False
    },
    {
        "id": "SAD_05",
        "category": "disappointed",
        "transcript": "私の確認不足だったのでしょうか…ご迷惑をおかけしました。",
        "acoustic_emotion": "remorse / sadness",
        "acoustic_conf": 0.87,
        "prosody": "quiet hesitant murmur, downcast cadence, apologetic sigh",
        "expected_sentiment": "disappointed_sad",
        "expected_conflict": False
    }
]

print(f"Running evaluation benchmark on {len(test_cases)} standardized cases...\n")
results = []
latencies = []

print(f"{'ID':<8} | {'Category':<12} | {'Expected':<18} | {'Predicted':<18} | {'Conf':<6} | {'Sarcasm(P)':<10} | {'Latency':<8} | {'Match'}")
print("-" * 95)

for case in test_cases:
    formatted_input = format_input(
        case["transcript"],
        case["acoustic_emotion"],
        case["acoustic_conf"],
        case["prosody"]
    )
    
    t0 = time.perf_counter()
    res = router.predict(formatted_input, CONSTANT_QUESTIONS, model="multilingual")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    latencies.append(elapsed_ms)
    
    pred_sent = res["answers"]["final_sentiment"]["choice"]
    pred_conf = res["answers"]["final_sentiment"]["answer_confidence"]
    pred_conflict_p = res["answers"]["is_sarcastic_or_conflicted"]["noul"]
    
    is_match = (pred_sent == case["expected_sentiment"])
    match_str = "PASS" if is_match else "FAIL"
    
    results.append({
        "id": case["id"],
        "category": case["category"],
        "transcript": case["transcript"],
        "expected": case["expected_sentiment"],
        "predicted": pred_sent,
        "confidence": pred_conf,
        "conflict_p": pred_conflict_p,
        "expected_conflict": case["expected_conflict"],
        "latency_ms": elapsed_ms,
        "match": is_match
    })
    
    print(f"{case['id']:<8} | {case['category']:<12} | {case['expected_sentiment']:<18} | {pred_sent:<18} | {pred_conf:.3f} | {pred_conflict_p:.3f}      | {elapsed_ms:>6.1f}ms | {match_str}")

# Metrics Summary
total = len(results)
correct = sum(1 for r in results if r["match"])
accuracy = (correct / total) * 100

# Category breakdown
categories = sorted(set(r["category"] for r in results))
cat_stats = {}
for cat in categories:
    sub = [r for r in results if r["category"] == cat]
    cat_correct = sum(1 for r in sub if r["match"])
    cat_stats[cat] = {
        "total": len(sub),
        "correct": cat_correct,
        "accuracy": (cat_correct / len(sub)) * 100,
        "avg_conf": sum(r["confidence"] for r in sub) / len(sub),
        "avg_conflict_p": sum(r["conflict_p"] for r in sub) / len(sub)
    }

# Sarcasm discrimination: compare conflict_p on sarcasm vs non-sarcasm
sarc_p = [r["conflict_p"] for r in results if r["expected_conflict"]]
non_sarc_p = [r["conflict_p"] for r in results if not r["expected_conflict"]]
avg_sarc_p = sum(sarc_p) / len(sarc_p) if sarc_p else 0.0
avg_non_sarc_p = sum(non_sarc_p) / len(non_sarc_p) if non_sarc_p else 0.0

latencies_sorted = sorted(latencies)
p50 = latencies_sorted[len(latencies)//2]
p95 = latencies_sorted[int(len(latencies)*0.95)]
mean_lat = sum(latencies) / len(latencies)

print("\n" + "=" * 70)
print("BENCHMARK PERFORMANCE SUMMARY")
print("=" * 70)
print(f"Overall Accuracy          : {correct}/{total} ({accuracy:.1f}%)")
print(f"Mean Latency (CPU)        : {mean_lat:.1f} ms")
print(f"Median (P50) Latency      : {p50:.1f} ms")
print(f"P95 Latency               : {p95:.1f} ms")
print(f"Min / Max Latency         : {min(latencies):.1f} ms / {max(latencies):.1f} ms")
print("-" * 70)
print("ACCURACY BY CATEGORY:")
for cat, st in cat_stats.items():
    print(f"  - {cat:<15}: {st['correct']}/{st['total']} ({st['accuracy']:.1f}%) | Avg Conf: {st['avg_conf']:.3f} | Avg Conflict P: {st['avg_conflict_p']:.3f}")
print("-" * 70)
print(f"Sarcasm Conflict Discrimination:")
print(f"  - Sarcastic samples (True conflict) avg conflict P : {avg_sarc_p:.4f}")
print(f"  - Non-sarcastic samples avg conflict P            : {avg_non_sarc_p:.4f}")
print("=" * 70)

# Save JSON dump for reference
with open("benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total": total,
            "accuracy": accuracy,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "cat_stats": cat_stats,
            "avg_sarc_p": avg_sarc_p,
            "avg_non_sarc_p": avg_non_sarc_p
        },
        "cases": results
    }, f, ensure_ascii=False, indent=2)

print("Detailed results saved to benchmark_results.json")
