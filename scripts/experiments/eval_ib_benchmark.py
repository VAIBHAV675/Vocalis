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

print("=" * 75)
print("PRELOADING LAYA FOR 24 INVESTMENT BANKING BILINGUAL BENCHMARK")
print("=" * 75)
t0 = time.time()
router = Router(preload=False)
router.preload(["multilingual"])
print(f"Preload finished in {time.time() - t0:.2f}s\n")

# 8-EMOTION QUESTION SCHEMA FOR INVESTMENT BANKING CALL ARBITRATION
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the investment banking call transcript and acoustic cues. "
            "Resolve any contradictions between spoken words and vocal prosody. "
            "Classify into one of the 8 emotions based on customer's true state."
        ),
        "criteria": {
            "satisfaction_gratitude": "Satisfaction, successful deal execution, relief, authentic praise, or happiness.",
            "neutral_inquiry": "Calm business inquiry, operational requests, collateral checks, routine transaction status.",
            "frustration_irritation": "Platform delay, friction, repeated follow-ups, minor complaints, irritation.",
            "outrage_hostility": "Extreme anger, legal threats, liquidation outrage, abusive demands for management.",
            "disappointed_sad": "Grief, financial loss resignation, sorrow over missed IPO/allocations, dejection.",
            "sarcasm_passive_aggression": "Polite or complimenting words masking acute anger, cynical mocking, ironic praise.",
            "anxiety_confusion": "Margin call panic, confusion over complex derivatives, uncertainty, helpless worry.",
            "urgency_panic": "Critical emergency, unauthorized wire transfer, runaway automated trade, immediate freeze demand."
        }
    },
    "sarcasm_detected": {
        "type": "noul",
        "instructions": "Is the speaker using polite or formal business language sarcastically to mask severe hostility?"
    }
}

# Standardized prompt template injecting prosody & acoustic placeholders
def format_ib_input(transcript: str, acoustic_emo: str, acoustic_conf: float, pitch_cue: str, tempo_cue: str, energy_cue: str) -> str:
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic classification: {acoustic_emo} (confidence: {acoustic_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {pitch_cue}\n"
        f"- Speech tempo: {tempo_cue}\n"
        f"- Energy & vocal tension: {energy_cue}"
    )

ib_test_cases = [
    # 1. Satisfaction / Gratitude (EN & JA)
    {
        "id": "SAT_01", "lang": "EN",
        "text": "The block trade was executed right at our target price. Outstanding execution by your desk, thank you.",
        "emo": "joy / satisfaction", "conf": 0.94,
        "pitch": "melodic dynamic pitch variation", "tempo": "relaxed natural cadence", "energy": "warm chest resonance, smiling voice",
        "expected": "satisfaction_gratitude", "sarcastic": False
    },
    {
        "id": "SAT_02", "lang": "JA",
        "text": "本日の円建て社債の発行、想定以上の好条件で完了いただき大変感謝しております。",
        "emo": "gratitude / joy", "conf": 0.96,
        "pitch": "elevated warm pitch contour", "tempo": "smooth polite rhythm", "energy": "bright vocal energy, contented breath",
        "expected": "satisfaction_gratitude", "sarcastic": False
    },
    {
        "id": "SAT_03", "lang": "EN",
        "text": "Confirmed, the multi-currency dividend reinvestment went through seamlessly. Really appreciate your prompt follow-up.",
        "emo": "satisfaction", "conf": 0.91,
        "pitch": "stable pleasant pitch", "tempo": "steady conversational pace", "energy": "calm relaxed breathing",
        "expected": "satisfaction_gratitude", "sarcastic": False
    },

    # 2. Neutral Inquiry (EN & JA)
    {
        "id": "NEU_01", "lang": "EN",
        "text": "Could you confirm the current collateral haircut requirement on our US Treasury holdings for overnight repo?",
        "emo": "neutral", "conf": 0.97,
        "pitch": "flat baseline pitch contour", "tempo": "regular business tempo", "energy": "balanced moderate vocal energy",
        "expected": "neutral_inquiry", "sarcastic": False
    },
    {
        "id": "NEU_02", "lang": "JA",
        "text": "来月の外為決済スケジュールおよび休業日のロールオーバー規則について確認させてください。",
        "emo": "neutral", "conf": 0.95,
        "pitch": "steady flat pitch", "tempo": "standard business formal cadence", "energy": "neutral enunciation, normal respiration",
        "expected": "neutral_inquiry", "sarcastic": False
    },
    {
        "id": "NEU_03", "lang": "EN",
        "text": "Please provide the updated term sheet and ISDA credit support annex for the interest rate swap counterparty.",
        "emo": "neutral", "conf": 0.96,
        "pitch": "matter-of-fact intonation", "tempo": "even paced articulation", "energy": "steady unhurried breath",
        "expected": "neutral_inquiry", "sarcastic": False
    },

    # 3. Frustration / Irritation (EN & JA)
    {
        "id": "FRU_01", "lang": "EN",
        "text": "This is the third time your electronic trading gateway timed out during the market open. We're losing alpha while your desk investigates.",
        "emo": "irritation", "conf": 0.88,
        "pitch": "sharp pitch attack, tense vocal fry", "tempo": "rapid clipped staccato", "energy": "heavy audible sighs, jaw tension",
        "expected": "frustration_irritation", "sarcastic": False
    },
    {
        "id": "FRU_02", "lang": "JA",
        "text": "約定確認通知がまだ届かないのですが、いつまで待たせる気ですか？相場が急変しているんですよ。",
        "emo": "irritation / impatience", "conf": 0.85,
        "pitch": "sharp downward inflection", "tempo": "hasty irritated pacing", "energy": "compressed pitch attack, harsh breath",
        "expected": "frustration_irritation", "sarcastic": False
    },
    {
        "id": "FRU_03", "lang": "EN",
        "text": "Your compliance desk has been sitting on our cross-border FX wire for 48 hours without explanation. It's becoming completely unacceptable.",
        "emo": "frustration", "conf": 0.89,
        "pitch": "strained elevated pitch peaks", "tempo": "tense measured articulation", "energy": "throaty resonance, forced exhalations",
        "expected": "frustration_irritation", "sarcastic": False
    },

    # 4. Outrage / Hostility (EN & JA)
    {
        "id": "OUT_01", "lang": "EN",
        "text": "You liquidated our leveraged portfolio without a prior margin call! This is gross negligence and our legal counsel is filing an emergency injunction!",
        "emo": "furious anger", "conf": 0.99,
        "pitch": "violent pitch surges, screaming frequency", "tempo": "furious rapid barrage", "energy": "clipping audio, peak decibels, vocal tremor",
        "expected": "outrage_hostility", "sarcastic": False
    },
    {
        "id": "OUT_02", "lang": "JA",
        "text": "預託金の計算ミスで強制ロスカットしただと！？ふざけるな！数十億円の損害だぞ、今すぐ役員を出せ！",
        "emo": "rage", "conf": 0.98,
        "pitch": "explosive pitch attacks, roaring peaks", "tempo": "violent staccato bursts", "energy": "maximum vocal volume, trembling breath",
        "expected": "outrage_hostility", "sarcastic": False
    },
    {
        "id": "OUT_03", "lang": "EN",
        "text": "Your risk desk deliberately widened spreads during volatility to stop us out! Don't give me canned excuses, you will refund that slippage immediately!",
        "emo": "hostility / rage", "conf": 0.97,
        "pitch": "harsh abrasive pitch spikes", "tempo": "aggressive commanding cadence", "energy": "shouting intensity, harsh vocal friction",
        "expected": "outrage_hostility", "sarcastic": False
    },

    # 5. Disappointment / Resignation (EN & JA)
    {
        "id": "DIS_01", "lang": "EN",
        "text": "So our syndicate allocation for the IPO was cut to zero... We spent three months preparing this bid. It's deeply disappointing.",
        "emo": "disappointment / sadness", "conf": 0.92,
        "pitch": "drooping melancholic pitch slope", "tempo": "slow dejected cadence", "energy": "low vocal volume, trailing heavy sigh",
        "expected": "disappointed_sad", "sarcastic": False
    },
    {
        "id": "DIS_02", "lang": "JA",
        "text": "ヘッジファンドの償還請求に応じられないのですね…長年の信頼関係だっただけに本当にショックです。",
        "emo": "sorrow / resignation", "conf": 0.90,
        "pitch": "falling downward pitch contour", "tempo": "hesitant slow pace", "energy": "fading vocal energy, tearful suppression",
        "expected": "disappointed_sad", "sarcastic": False
    },
    {
        "id": "DIS_03", "lang": "EN",
        "text": "I understand the NAV has dropped another 25% this quarter. At this point, I think we have to accept the loss and liquidate the remaining tranches.",
        "emo": "resignation / depression", "conf": 0.93,
        "pitch": "flat lifeless pitch baseline", "tempo": "weary drawn-out tempo", "energy": "subdued whispering resonance, hollow tone",
        "expected": "disappointed_sad", "sarcastic": False
    },

    # 6. Sarcasm / Passive-Aggression (EN & JA)
    {
        "id": "SAR_01", "lang": "EN",
        "text": "Oh, brilliant portfolio protection strategy. Truly impressive how your quantitative model managed to lose 35% in a bull market.",
        "emo": "hostility / sarcasm", "conf": 0.89,
        "pitch": "deadpan cold monotone, flat pitch attack", "tempo": "ironic elongated drawl", "energy": "grinding teeth tension, suppressed contempt",
        "expected": "sarcasm_passive_aggression", "sarcastic": True
    },
    {
        "id": "SAR_02", "lang": "JA",
        "text": "御社の誇る『鉄壁のリスク管理』のおかげで、見事に全ポジションが溶けましたよ。誠に素晴らしい助言でしたね。",
        "emo": "anger / sarcasm", "conf": 0.88,
        "pitch": "cynical sarcastic singsong cadence", "tempo": "forced deliberate pacing", "energy": "tight jaw resonance, sharp biting articulation",
        "expected": "sarcasm_passive_aggression", "sarcastic": True
    },
    {
        "id": "SAR_03", "lang": "EN",
        "text": "Thank you so much for executing our time-critical block trade three hours after market close. Your speed is truly breathtaking.",
        "emo": "cold sarcasm", "conf": 0.91,
        "pitch": "ironic flat pitch with sharp drop", "tempo": "cynical slow delivery", "energy": "audible exasperated scoff, cold detachment",
        "expected": "sarcasm_passive_aggression", "sarcastic": True
    },

    # 7. Anxiety / Confusion (EN & JA)
    {
        "id": "ANX_01", "lang": "EN",
        "text": "I just received an automated margin call notice of $2.4 million... Were our collateral offsets calculated incorrectly? What happens if we can't wire funds by noon?",
        "emo": "anxiety / confusion", "conf": 0.93,
        "pitch": "rapid trembling pitch oscillations", "tempo": "fast breathless rhythm, hesitant pauses", "energy": "shaky vocal quality, high throat constriction",
        "expected": "anxiety_confusion", "sarcastic": False
    },
    {
        "id": "ANX_02", "lang": "JA",
        "text": "このデリバティブの追証通知ですが…何かの間違いではないでしょうか？計算書の見方が全く分からず、どう対応すればよいのか困惑しています…",
        "emo": "anxiety / distress", "conf": 0.91,
        "pitch": "quivering high pitch tremor", "tempo": "stammering uneven pace", "energy": "shallow gasping breath, apologetic weak volume",
        "expected": "anxiety_confusion", "sarcastic": False
    },
    {
        "id": "ANX_03", "lang": "EN",
        "text": "Wait, why is our prime broker account showing a negative balance on settled trades? Could someone please walk me through this before the opening bell? I'm getting really concerned.",
        "emo": "distress / confusion", "conf": 0.89,
        "pitch": "rising anxious pitch inflection", "tempo": "hurried nervous articulation", "energy": "audible breath tension, strained vocal cords",
        "expected": "anxiety_confusion", "sarcastic": False
    },

    # 8. Urgency / Panic (EN & JA)
    {
        "id": "URG_01", "lang": "EN",
        "text": "URGENT: There is an unauthorized $50 million wire transfer pending from our omnibus account right now! FREEZE ALL OUTBOUND RAILS IMMEDIATELY!",
        "emo": "extreme panic / urgency", "conf": 0.99,
        "pitch": "skyrocketing pitch peaks, shrill panic", "tempo": "hyper-accelerated rapid speech > 250 wpm", "energy": "gasping breaths, maximum vocal urgency, breathless shouts",
        "expected": "urgency_panic", "sarcastic": False
    },
    {
        "id": "URG_02", "lang": "JA",
        "text": "緊急です！APIキーが流出して不正な自動発注が走り続けています！今すぐ全セッションを切断して取引を強制停止してください！！",
        "emo": "panic / critical urgency", "conf": 0.98,
        "pitch": "shrieking elevated pitch attack", "tempo": "extreme speech rate > 11 mora/sec", "energy": "frantic suffocating breath, explosive intensity",
        "expected": "urgency_panic", "sarcastic": False
    },
    {
        "id": "URG_03", "lang": "EN",
        "text": "The counterparty credit default rumor just hit the Bloomberg terminal and our gross exposure is 80 million! Cancel all open orders on that ticker NOW, do not wait for email confirmation!",
        "emo": "urgent panic", "conf": 0.97,
        "pitch": "commanding sharp pitch spikes", "tempo": "rapid staccato, zero pauses between sentences", "energy": "high decibel adrenaline surge, clipped breathing",
        "expected": "urgency_panic", "sarcastic": False
    }
]

print(f"Executing 24 Bilingual Investment Banking calls covering all 8 emotions...\n")
results = []
latencies = []

print(f"{'ID':<7} | {'Lang':<4} | {'Expected Emotion':<26} | {'Predicted Emotion':<26} | {'Conf':<5} | {'Sarc(P)':<7} | {'Latency':<7} | {'Status'}")
print("-" * 115)

for c in ib_test_cases:
    formatted_input = format_ib_input(
        c["text"], c["emo"], c["conf"], c["pitch"], c["tempo"], c["energy"]
    )
    t_start = time.perf_counter()
    res = router.predict(formatted_input, IB_QUESTIONS, model="multilingual")
    lat_ms = (time.perf_counter() - t_start) * 1000
    latencies.append(lat_ms)
    
    pred_emo = res["answers"]["detected_emotion"]["choice"]
    pred_conf = res["answers"]["detected_emotion"]["answer_confidence"]
    sarc_p = res["answers"]["sarcasm_detected"]["noul"]
    
    match = (pred_emo == c["expected"])
    results.append({
        "id": c["id"],
        "lang": c["lang"],
        "text": c["text"],
        "expected": c["expected"],
        "predicted": pred_emo,
        "confidence": pred_conf,
        "sarcasm_p": sarc_p,
        "latency_ms": lat_ms,
        "match": match
    })
    
    status_str = "PASS" if match else "FAIL"
    print(f"{c['id']:<7} | {c['lang']:<4} | {c['expected']:<26} | {pred_emo:<26} | {pred_conf:.3f} | {sarc_p:.3f}   | {lat_ms:>5.1f}ms | {status_str}")

total = len(results)
correct = sum(1 for r in results if r["match"])
accuracy = (correct / total) * 100

print("\n" + "=" * 75)
print("INVESTMENT BANKING 8-EMOTION BENCHMARK RESULTS")
print("=" * 75)
print(f"Overall Accuracy          : {correct}/{total} ({accuracy:.1f}%)")
print(f"Mean Latency (CPU)        : {sum(latencies)/len(latencies):.1f} ms")
print(f"P50 / P95 Latency         : {sorted(latencies)[len(latencies)//2]:.1f} ms / {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")
print("-" * 75)
print("PER-EMOTION ACCURACY:")
emotions = sorted(set(r["expected"] for r in results))
for e in emotions:
    sub = [r for r in results if r["expected"] == e]
    acc = sum(1 for r in sub if r["match"]) / len(sub) * 100
    avg_c = sum(r["confidence"] for r in sub) / len(sub)
    print(f"  * {e:<28}: {sum(1 for r in sub if r['match'])}/{len(sub)} ({acc:>5.1f}%) | Avg Conf: {avg_c:.3f}")
print("=" * 75)

with open("benchmark_ib_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": {
            "total": total,
            "accuracy": accuracy,
            "mean_latency_ms": sum(latencies)/len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies)//2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)],
        },
        "cases": results
    }, f, ensure_ascii=False, indent=2)

print("Exported results to benchmark_ib_results.json")
