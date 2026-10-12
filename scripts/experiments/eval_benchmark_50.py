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

print("=" * 70)
print("PRELOADING LAYA FOR 50-SAMPLE BENCHMARK")
print("=" * 70)
t0 = time.time()
router = Router(preload=False)
router.preload(["multilingual"])
print(f"Preload finished in {time.time() - t0:.2f}s\n")

# Constant prompt template & questions
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

def format_input(transcript: str, acoustic_emotion: str, acoustic_conf: float, prosody: str) -> str:
    return (
        f"Speaker text: 「{transcript}」\n"
        f"Acoustic vocal tone: {acoustic_emotion} (confidence: {acoustic_conf:.0%})\n"
        f"Prosody cues: {prosody}"
    )

# 50 Detailed Japanese Scenarios
test_cases_50 = [
    # --- Group A: Sarcasm & Masked Aggression (10 cases) ---
    {"id": "SARC_01", "cat": "sarcasm", "text": "ご丁寧な対応、大変恐れ入ります。", "emo": "anger / sarcasm", "conf": 0.88, "pros": "cold monotone, sharp pitch drop, tense vocal fry", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_02", "cat": "sarcasm", "text": "あっそう、素晴らしいマニュアル通りのご対応ですね。", "emo": "contempt / anger", "conf": 0.91, "pros": "cynical scoff, elevated pitch attack, aggressive pace", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_03", "cat": "sarcasm", "text": "分かりました、御社のご都合が最優先ということですもんね。", "emo": "irritation", "conf": 0.84, "pros": "flat passive-aggressive drawl, heavy sigh", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_04", "cat": "sarcasm", "text": "お役に立てて光栄でした、本当に。", "emo": "cold sarcasm", "conf": 0.87, "pros": "strained vocal cords, tight jaw resonance, low pitch", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_05", "cat": "sarcasm", "text": "ええ、何一つ問題ありませんよ。全く。", "emo": "anger", "conf": 0.83, "pros": "clipped aggressive staccato, breath expulsion", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_06", "cat": "sarcasm", "text": "ご丁寧に長々と説明してくださって恐縮です。", "emo": "exasperation / anger", "conf": 0.86, "pros": "audible eye-roll sigh, sharp vocal tension", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_07", "cat": "sarcasm", "text": "素晴らしいシステムですね、二度と使いませんけど。", "emo": "hostility", "conf": 0.89, "pros": "snapping pitch inflection, tight throat resonance", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_08", "cat": "sarcasm", "text": "貴重なお時間をいただき、心より感謝申し上げます。", "emo": "anger / hostility", "conf": 0.85, "pros": "ironic singsong cadence, grinding teeth tension", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_09", "cat": "sarcasm", "text": "結構です。もうお話しすることはございません。", "emo": "cold anger", "conf": 0.92, "pros": "subdued chilling volume, dead flat pitch, abrupt breath", "exp": "frustrated_hostile", "conflicted": True},
    {"id": "SARC_10", "cat": "sarcasm", "text": "お忙しいのにわざわざお返事いただき申し訳ありませんでしたね。", "emo": "passive-aggression", "conf": 0.82, "pros": "sharp pitch tremor, sarcastic elongation of syllables", "exp": "frustrated_hostile", "conflicted": True},

    # --- Group B: Genuine Satisfaction & Joy (10 cases) ---
    {"id": "POS_01", "cat": "positive", "text": "親切に教えていただき本当に助かりました！", "emo": "joy / gratitude", "conf": 0.95, "pros": "smiling voice, high pitch variation, warm cadence", "exp": "positive", "conflicted": False},
    {"id": "POS_02", "cat": "positive", "text": "すぐ解決できて安心しました。ありがとうございます！", "emo": "relief / joy", "conf": 0.93, "pros": "relaxed exhalation, bright resonance, animated tempo", "exp": "positive", "conflicted": False},
    {"id": "POS_03", "cat": "positive", "text": "とても丁寧なサポートで感動しました。", "emo": "joy / admiration", "conf": 0.96, "pros": "soft warm energy, lilting melodic intonation", "exp": "positive", "conflicted": False},
    {"id": "POS_04", "cat": "positive", "text": "担当の山田様によろしくお伝えください。大満足です。", "emo": "satisfaction", "conf": 0.90, "pros": "steady pleasant cadence, open chest resonance", "exp": "positive", "conflicted": False},
    {"id": "POS_05", "cat": "positive", "text": "案内がすごく分かりやすかったです！また利用しますね。", "emo": "enthusiasm", "conf": 0.94, "pros": "upbeat cadence, dynamic pitch peaks, lively energy", "exp": "positive", "conflicted": False},
    {"id": "POS_06", "cat": "positive", "text": "おかげさまで設定がすべて完了しました！", "emo": "relief / happiness", "conf": 0.92, "pros": "bright pitch lift, cheerful chuckling tone", "exp": "positive", "conflicted": False},
    {"id": "POS_07", "cat": "positive", "text": "いつも迅速にご対応いただき感謝しております。", "emo": "gratitude", "conf": 0.89, "pros": "calm warm tone, smooth pacing, balanced energy", "exp": "positive", "conflicted": False},
    {"id": "POS_08", "cat": "positive", "text": "こんなに早く返信がもらえるとは思っていませんでした。嬉しいです！", "emo": "delight", "conf": 0.93, "pros": "elevated pitch attack, rapid excited tempo", "exp": "positive", "conflicted": False},
    {"id": "POS_09", "cat": "positive", "text": "完璧です。求めていた通りの回答でした。", "emo": "satisfaction", "conf": 0.91, "pros": "firm confident cadence, contented breathing", "exp": "positive", "conflicted": False},
    {"id": "POS_10", "cat": "positive", "text": "ご親切に寄り添っていただき本当に救われました。", "emo": "deep gratitude", "conf": 0.95, "pros": "warm emotional trembling, soft sincere vocal energy", "exp": "positive", "conflicted": False},

    # --- Group C: Neutral Business Inquiries (10 cases) ---
    {"id": "NEUT_01", "cat": "neutral", "text": "アカウントのログインIDを変更したいのですが手順は？", "emo": "neutral", "conf": 0.96, "pros": "even conversational pace, flat baseline pitch", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_02", "cat": "neutral", "text": "支払方法をクレジットカードから銀行振込に変えられますか。", "emo": "neutral", "conf": 0.94, "pros": "standard business cadence, calm matter-of-fact tone", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_03", "cat": "neutral", "text": "契約更新の案内通知はいつ頃届きますでしょうか。", "emo": "neutral", "conf": 0.93, "pros": "steady pitch, normal speaking tempo, balanced energy", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_04", "cat": "neutral", "text": "プランAとプランBの差分について教えてください。", "emo": "neutral", "conf": 0.95, "pros": "clear articulation, neutral enquiry intonation", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_05", "cat": "neutral", "text": "領収書の再発行をお願いしたいのですが可能ですか。", "emo": "neutral", "conf": 0.92, "pros": "formal polite cadence, steady breathing", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_06", "cat": "neutral", "text": "次回メンテナンスの日程と影響範囲を確認させてください。", "emo": "neutral", "conf": 0.97, "pros": "standard monotone inquiry, balanced pitch", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_07", "cat": "neutral", "text": "登録メールアドレスに確認コードが届いているか確認します。", "emo": "neutral", "conf": 0.91, "pros": "even conversational tone, quiet neutral pacing", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_08", "cat": "neutral", "text": "利用規約の第5条に関する補足資料はありますか。", "emo": "neutral", "conf": 0.94, "pros": "straightforward query, flat pitch contour", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_09", "cat": "neutral", "text": "解約手続きの締め切り日は毎月何日になりますか。", "emo": "neutral", "conf": 0.93, "pros": "calm matter-of-fact cadence, regular breath", "exp": "neutral", "conflicted": False},
    {"id": "NEUT_10", "cat": "neutral", "text": "セミナーの参加者枠にまだ空きはありますでしょうか。", "emo": "neutral", "conf": 0.95, "pros": "inquisitive neutral tone, moderate tempo", "exp": "neutral", "conflicted": False},

    # --- Group D: Open Hostility & Anger (10 cases) ---
    {"id": "HOST_01", "cat": "hostile", "text": "いつまで待たせる気だ！いい加減にしろ！", "emo": "furious anger", "conf": 0.98, "pros": "loud shouting, clipping audio, violent pitch surge", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_02", "cat": "hostile", "text": "お前の名前は何だ！上司に代われ今すぐ！", "emo": "anger", "conf": 0.96, "pros": "harsh sharp pitch attack, shaking tense jaw", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_03", "cat": "hostile", "text": "金だけ取って動かないとか完全に詐欺だろうが！", "emo": "rage", "conf": 0.97, "pros": "explosive volume, rapid aggressive cadence", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_04", "cat": "hostile", "text": "何回同じ説明をさせるんだ！メモも取ってないのか！", "emo": "anger / exasperation", "conf": 0.94, "pros": "snapping pitch peaks, harsh abrasive vocal fry", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_05", "cat": "hostile", "text": "使い物にならないゴミシステムだな。金返せ！", "emo": "disgust / anger", "conf": 0.95, "pros": "spitting enunciation, heavy abrasive pressure", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_06", "cat": "hostile", "text": "ふざけた言い訳ばかり並べるな！誠意を見せろ！", "emo": "hostility", "conf": 0.93, "pros": "bellowing voice, heavy breath tremors", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_07", "cat": "hostile", "text": "そっちのミスでこっちの業務が止まってるんだぞ！責任取れ！", "emo": "rage / urgency", "conf": 0.97, "pros": "intense volume spikes, rapid suffocating pacing", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_08", "cat": "hostile", "text": "謝って済む問題じゃないんだよ！どう補償するんだ！", "emo": "anger", "conf": 0.96, "pros": "aggressive staccato, harsh throat tension", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_09", "cat": "hostile", "text": "もう我慢の限界だ。消費者センターに通報するからな！", "emo": "vindictive anger", "conf": 0.94, "pros": "sharp venomous cadence, elevated pitch plateau", "exp": "frustrated_hostile", "conflicted": False},
    {"id": "HOST_10", "cat": "hostile", "text": "二度と顔も見たくない。今すぐ全額返金の手続きをしろ！", "emo": "anger / disgust", "conf": 0.98, "pros": "loud commanding roar, abrupt furious termination", "exp": "frustrated_hostile", "conflicted": False},

    # --- Group E: Disappointment, Resignation & Sadness (10 cases) ---
    {"id": "SAD_01", "cat": "disappointed", "text": "ずっと楽しみにしていたのに…本当に悲しいです。", "emo": "sadness", "conf": 0.92, "pros": "downcast pitch glide, trembling voice, tearful intake", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_02", "cat": "disappointed", "text": "どうやっても元には戻らないんですね…ショックです。", "emo": "despair / sadness", "conf": 0.90, "pros": "fading vocal energy, heavy sorrowful exhalation", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_03", "cat": "disappointed", "text": "期待が大きかっただけに、とても残念に思います。", "emo": "disappointment", "conf": 0.88, "pros": "subdued slow tempo, flat dejected tone", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_04", "cat": "disappointed", "text": "そうですか…無理を言って申し訳ありませんでした。", "emo": "resignation / guilt", "conf": 0.86, "pros": "weak volume, trailing whispery cadence, heavy sigh", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_05", "cat": "disappointed", "text": "もうどうしようもないんですね…分かりました、諦めます。", "emo": "resignation", "conf": 0.93, "pros": "slow exhausted cadence, lifeless flat resonance", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_06", "cat": "disappointed", "text": "私のやり方が悪かったのかもしれません…すみませんでした。", "emo": "self-reproach / sadness", "conf": 0.89, "pros": "quiet hesitant murmur, breaking vocal quality", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_07", "cat": "disappointed", "text": "長年愛用していたサービスだっただけに、改悪されて寂しいです。", "emo": "sorrow / disappointment", "conf": 0.87, "pros": "melancholic intonation, slow wistful cadence", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_08", "cat": "disappointed", "text": "これ以上の対応は難しいのですね…途方に暮れています。", "emo": "helplessness / sadness", "conf": 0.91, "pros": "trembling pitch, shaky breath pauses, low volume", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_09", "cat": "disappointed", "text": "楽しみにしていたイベントに行けなくなってしまいました…", "emo": "disappointment", "conf": 0.94, "pros": "drooping pitch, tearful choking quality, subdued resonance", "exp": "disappointed_sad", "conflicted": False},
    {"id": "SAD_10", "cat": "disappointed", "text": "何をやってもうまくいかなくて…疲れてしまいました。", "emo": "exhaustion / depression", "conf": 0.90, "pros": "very slow monotonic tempo, fading exhalations", "exp": "disappointed_sad", "conflicted": False}
]

print(f"Starting execution of 50 standardized test cases...\n")
results = []
latencies = []

print(f"{'ID':<8} | {'Category':<12} | {'Expected':<18} | {'Predicted':<18} | {'Conf':<6} | {'Sarc(P)':<8} | {'Latency':<8} | {'Status'}")
print("-" * 94)

for c in test_cases_50:
    formatted = format_input(c["text"], c["emo"], c["conf"], c["pros"])
    t_start = time.perf_counter()
    res = router.predict(formatted, CONSTANT_QUESTIONS, model="multilingual")
    lat_ms = (time.perf_counter() - t_start) * 1000
    latencies.append(lat_ms)
    
    pred_s = res["answers"]["final_sentiment"]["choice"]
    pred_c = res["answers"]["final_sentiment"]["answer_confidence"]
    pred_conflict = res["answers"]["is_sarcastic_or_conflicted"]["noul"]
    
    match = (pred_s == c["exp"])
    results.append({
        "id": c["id"],
        "category": c["cat"],
        "text": c["text"],
        "expected": c["exp"],
        "predicted": pred_s,
        "confidence": pred_c,
        "conflict_p": pred_conflict,
        "expected_conflict": c["conflicted"],
        "latency_ms": lat_ms,
        "match": match
    })
    
    status_str = "PASS" if match else "FAIL"
    print(f"{c['id']:<8} | {c['cat']:<12} | {c['exp']:<18} | {pred_s:<18} | {pred_c:.3f} | {pred_conflict:.3f}   | {lat_ms:>6.1f}ms | {status_str}")

# Summary Stats
total = len(results)
correct = sum(1 for r in results if r["match"])
accuracy = (correct / total) * 100

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
        "avg_sarc_p": sum(r["conflict_p"] for r in sub) / len(sub)
    }

sarc_true = [r["conflict_p"] for r in results if r["expected_conflict"]]
sarc_false = [r["conflict_p"] for r in results if not r["expected_conflict"]]
avg_sarc_true = sum(sarc_true) / len(sarc_true) if sarc_true else 0.0
avg_sarc_false = sum(sarc_false) / len(sarc_false) if sarc_false else 0.0

lat_sorted = sorted(latencies)
p50 = lat_sorted[len(latencies)//2]
p95 = lat_sorted[int(len(latencies)*0.95)]
mean_lat = sum(latencies) / len(latencies)

print("\n" + "=" * 70)
print("50-SAMPLE BENCHMARK PERFORMANCE REPORT")
print("=" * 70)
print(f"Total Evaluated Samples   : {total}")
print(f"Overall Accuracy          : {correct}/{total} ({accuracy:.1f}%)")
print(f"Mean Latency (CPU)        : {mean_lat:.1f} ms")
print(f"Median (P50) Latency      : {p50:.1f} ms")
print(f"P95 Latency               : {p95:.1f} ms")
print(f"Min / Max Latency         : {min(latencies):.1f} ms / {max(latencies):.1f} ms")
print("-" * 70)
print("CATEGORY PERFORMANCE BREAKDOWN:")
for cat, st in cat_stats.items():
    print(f"  * {cat:<15}: {st['correct']:>2}/{st['total']:>2} ({st['accuracy']:>5.1f}%) | Avg Conf: {st['avg_conf']:.3f} | Sarcasm P: {st['avg_sarc_p']:.3f}")
print("-" * 70)
print(f"Sarcasm Discrimination Contrast:")
print(f"  * Truly Conflicted/Sarcastic Cases  : {avg_sarc_true:.4f}")
print(f"  * Non-Sarcastic Regular Cases       : {avg_sarc_false:.4f}")
print("=" * 70)

# Save JSON dump
with open("benchmark_50_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total": total,
            "accuracy": accuracy,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "cat_stats": cat_stats,
            "avg_sarc_true": avg_sarc_true,
            "avg_sarc_false": avg_sarc_false
        },
        "cases": results
    }, f, ensure_ascii=False, indent=2)

print("Full results exported to benchmark_50_results.json")
