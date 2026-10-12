import os
import sys
import time
import json
import gc
from pathlib import Path

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

# Load .env & flags
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

# CUDA check
import torch
print("=" * 80)
print(f"DEVICE: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print("PIPELINE: TRI-MODAL SPEECH INTELLIGENCE (PROSODY + SER FOUNDATION + LAYA mmBERT)")
print("BENCHMARK: 24 ENGLISH-ONLY & 24 JAPANESE-ONLY INVESTMENT BANKING CALLS (48 TOTAL)")
print("=" * 80)

from laya import Router

router = Router(preload=False)
t0 = time.time()
router.preload(["multilingual"])
print(f"Laya mmBERT preloaded in {time.time() - t0:.2f}s\n")

# 8-EMOTION QUESTION SCHEMA FOR INVESTMENT BANKING ARBITRATION
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the caller transcript and acoustic prosody hints. "
            "Resolve any contradictions between spoken words and vocal tone (e.g. sarcasm, masked anger). "
            "Classify into one of the 8 canonical emotions based on the customer's true mental and emotional state."
        ),
        "criteria": {
            "satisfaction_gratitude": "Delight, genuine satisfaction, heartfelt gratitude, relief, happiness, warm appreciation.",
            "neutral_inquiry": "Calm routine business inquiry, objective questions, matter-of-fact tone, operational transaction queries.",
            "frustration_irritation": "Annoyance, impatient complaints, repeated friction, irritation, vocal exasperation.",
            "outrage_hostility": "Extreme fury, aggressive hostility, shouting, legal threats, intense confrontation.",
            "disappointed_sad": "Sorrow, deep sadness, grief, emotional dejection, heavy sighing, depressed hopeless tone.",
            "sarcasm_passive_aggression": "Ironic praise, mocking cynicism, polite words concealing sharp anger and contempt.",
            "anxiety_confusion": "Nervous panic, trembling uncertainty, fear, worried distress, helpless confusion.",
            "urgency_panic": "Critical crisis, urgent emergency, immediate stop or freeze demand, frantic alarm."
        }
    },
    "sarcasm_detected": {
        "type": "noul",
        "instructions": "Is the speaker using polite or formal business language sarcastically to mask severe hostility?"
    }
}

# Clean prompt template injecting acoustic prosodic placeholders into Laya
def format_prompt(transcript: str, ser_prior: str, ser_conf: float, pitch_cue: str, tempo_cue: str, energy_cue: str, voice_quality: str) -> str:
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic speech emotion prior: {ser_prior} (confidence: {ser_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {pitch_cue}\n"
        f"- Speech tempo: {tempo_cue}\n"
        f"- Energy & vocal tension: {energy_cue}\n"
        f"- Voice quality: {voice_quality}"
    )

# -------------------------------------------------------------------------
# 24 ENGLISH-ONLY CALLS (3 CALLS PER EMOTION ACROSS ALL 8 EMOTIONS)
# -------------------------------------------------------------------------
EN_CALLS = [
    # 1. satisfaction_gratitude
    {
        "id": "EN_SAT_01", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "The syndicated debt facility closed exactly at our target spread of 145 basis points. Outstanding execution by your leveraged finance team, thank you so much.",
        "ser": "joy / satisfaction", "conf": 0.95,
        "pitch": "142 Hz (warm, melodic dynamic rise on key verbs)", "tempo": "4.2 syll/sec (relaxed natural cadence)", "energy": "68 dB (warm chest resonance)", "vq": "low jitter 0.4%, smiling phonation"
    },
    {
        "id": "EN_SAT_02", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "Confirmed, the multi-currency dividend repatriation went through without a hitch. Really appreciate your operations desk turning that around so fast.",
        "ser": "satisfaction", "conf": 0.92,
        "pitch": "135 Hz (smooth pleasant contour)", "tempo": "4.5 syll/sec (steady comfortable pace)", "energy": "65 dB (calm relaxed breathing)", "vq": "harmonic stability, zero glottal tension"
    },
    {
        "id": "EN_SAT_03", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "We just saw the allocation notification for the sovereign green bond. We got our full requested book size, which is tremendous news for our fund.",
        "ser": "relief / joy", "conf": 0.96,
        "pitch": "158 Hz (excited elevated terminal contour)", "tempo": "5.1 syll/sec (enthusiastic energetic tempo)", "energy": "72 dB (bright resonant vocal output)", "vq": "resonant vocal cords, genuine laughter token"
    },

    # 2. neutral_inquiry
    {
        "id": "EN_NEU_01", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "Good morning. Could you please provide the updated haircut percentage on our AAA-rated euro collateral basket for the overnight repo facility?",
        "ser": "neutral", "conf": 0.98,
        "pitch": "118 Hz (flat monotonic pitch)", "tempo": "4.0 syll/sec (standard business pace)", "energy": "60 dB (even baseline intensity)", "vq": "normal modal voice, zero jitter spikes"
    },
    {
        "id": "EN_NEU_02", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "We are reconciling our Q3 settlement logs and need the exact timestamp when trade reference TXN-9482 cleared through Euroclear.",
        "ser": "neutral", "conf": 0.97,
        "pitch": "122 Hz (steady horizontal contour)", "tempo": "4.3 syll/sec (clear measured enunciation)", "energy": "62 dB (steady ambient volume)", "vq": "standard modal phonation"
    },
    {
        "id": "EN_NEU_03", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "Can you confirm whether the ex-dividend date on the senior floating rate notes falls on this Thursday or Friday?",
        "ser": "neutral / calm", "conf": 0.95,
        "pitch": "120 Hz (balanced neutral inflection)", "tempo": "4.1 syll/sec (unhurried factual pacing)", "energy": "61 dB (stable controlled volume)", "vq": "no vocal fry, clean acoustic spectrum"
    },

    # 3. frustration_irritation
    {
        "id": "EN_FRU_01", "expected": "frustration_irritation", "sarcastic": False,
        "text": "This is the third time today your FIX API gateway has disconnected during market open. We have over 12 million dollars in pending orders sitting unrouted.",
        "ser": "irritation", "conf": 0.89,
        "pitch": "165 Hz (sharp elevated pitch on 'third time')", "tempo": "5.8 syll/sec (fast clipped cadence)", "energy": "74 dB (elevated volume, heavy sigh)", "vq": "jitter 1.8%, audible exasperated exhale"
    },
    {
        "id": "EN_FRU_02", "expected": "frustration_irritation", "sarcastic": False,
        "text": "I was promised the ISDA Master Agreement amendment would be executed by 2 PM. It is now 4:30 PM and my compliance desk is blocking our hedge.",
        "ser": "frustration", "conf": 0.91,
        "pitch": "158 Hz (strained pressed vocal cords)", "tempo": "5.2 syll/sec (abrupt pauses, tense rhythm)", "energy": "71 dB (intermittent loud emphasis)", "vq": "vocal tract tension, throat clearing"
    },
    {
        "id": "EN_FRU_03", "expected": "frustration_irritation", "sarcastic": False,
        "text": "Your back-office keeps sending margin discrepancy notices for accounts we closed two weeks ago. Can someone please fix your database records?",
        "ser": "annoyance", "conf": 0.88,
        "pitch": "150 Hz (downward irritated pitch glides)", "tempo": "4.8 syll/sec (irritated rhythmic stress)", "energy": "69 dB (tense mid-register)", "vq": "creaky voice on final syllables"
    },

    # 4. outrage_hostility
    {
        "id": "EN_OUT_01", "expected": "outrage_hostility", "sarcastic": False,
        "text": "You liquidated our collateral position with zero grace period! That is a blatant breach of our prime brokerage covenants! Get your head of legal on this phone right now!",
        "ser": "anger / rage", "conf": 0.98,
        "pitch": "285 Hz (extreme shrill pitch ceiling)", "tempo": "6.8 syll/sec (rapid explosive outburst)", "energy": "88 dB (near-clipping acoustic saturation)", "vq": "severe jitter 4.2%, harsh turbulent airflow"
    },
    {
        "id": "EN_OUT_02", "expected": "outrage_hostility", "sarcastic": False,
        "text": "You cost our partnership 4.5 million dollars due to your unauthorized trade cancellation! We are filing an emergency FINRA complaint and injunction this afternoon!",
        "ser": "outrage / hostility", "conf": 0.97,
        "pitch": "260 Hz (erratic violent pitch modulation)", "tempo": "6.4 syll/sec (aggressive staccato shouting)", "energy": "85 dB (deafening peak volume)", "vq": "glottalized explosive consonants, vocal strain"
    },
    {
        "id": "EN_OUT_03", "expected": "outrage_hostility", "sarcastic": False,
        "text": "Do not dare tell me to calm down! Your desk engaged in market manipulation on our block order! I will make sure every managing director on your board knows about this fraud!",
        "ser": "fury", "conf": 0.99,
        "pitch": "295 Hz (uncontrolled high-frequency screech)", "tempo": "7.1 syll/sec (furious breathless pace)", "energy": "91 dB (redlining microphone audio)", "vq": "extreme vocal perturbation, trembling cords"
    },

    # 5. disappointed_sad
    {
        "id": "EN_DIS_01", "expected": "disappointed_sad", "sarcastic": False,
        "text": "So we ended up getting zero shares in the technology IPO bookbuild after committing sixty million dollars... That pretty much ruins our entire Q4 performance strategy.",
        "ser": "sadness / dejection", "conf": 0.92,
        "pitch": "88 Hz (deep depressed pitch floor, monotone drop)", "tempo": "2.8 syll/sec (lethargic, dragged speaking rate)", "energy": "48 dB (subdued hollow vocal intensity)", "vq": "high breathiness, prolonged somber silence"
    },
    {
        "id": "EN_DIS_02", "expected": "disappointed_sad", "sarcastic": False,
        "text": "The junior debt tranche took a full 70% haircut in restructuring. We worked on this turnaround for two years, and now it's basically all gone.",
        "ser": "grief / disappointment", "conf": 0.94,
        "pitch": "92 Hz (faltering downward pitch trajectory)", "tempo": "3.1 syll/sec (halting heavy delivery)", "energy": "50 dB (quiet dejected voice)", "vq": "audible breathy sigh, vocal tremor"
    },
    {
        "id": "EN_DIS_03", "expected": "disappointed_sad", "sarcastic": False,
        "text": "I see the final marks on our structured credit portfolio. A twenty-five percent drawdown in a single month... There's really nothing more to say.",
        "ser": "despair / sadness", "conf": 0.91,
        "pitch": "85 Hz (flat low-frequency rumble)", "tempo": "2.6 syll/sec (drained exhausted cadence)", "energy": "46 dB (barely audible whispery timbre)", "vq": "attenuated upper harmonics, resigned tone"
    },

    # 6. sarcasm_passive_aggression
    {
        "id": "EN_SAR_01", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "Oh, what an absolutely brilliant trade execution! Slipping our fill price by forty-five pips on a liquid pair is truly top-tier Wall Street investment banking at its finest.",
        "ser": "anger / hostility (acoustic prosody)", "conf": 0.91,
        "pitch": "240 Hz (exaggerated theatrical pitch contour)", "tempo": "3.2 syll/sec (elongated sarcastic vowels)", "energy": "78 dB (sharp bite on 'brilliant')", "vq": "cynical pressed phonation, mocking cadence"
    },
    {
        "id": "EN_SAR_02", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "Thank you so much for taking three full business days to answer an urgent collateral email. Your dedication to customer service is truly overwhelming.",
        "ser": "contempt / sarcasm", "conf": 0.94,
        "pitch": "215 Hz (hyper-formal melodic mockery)", "tempo": "3.4 syll/sec (ironic staccato drawl)", "energy": "73 dB (tense suppressed fury)", "vq": "acidic undertone, forced false smile"
    },
    {
        "id": "EN_SAR_03", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "It was wonderful of your risk department to automatically lock our account right before the FOMC rate decision. Just spectacular timing as always.",
        "ser": "passive aggressive / anger", "conf": 0.93,
        "pitch": "230 Hz (artificial singsong pitch peaks)", "tempo": "3.5 syll/sec (deliberate cynical emphasis)", "energy": "75 dB (piercing sarcastic consonants)", "vq": "tense glottal constriction, contemptuous bite"
    },

    # 7. anxiety_confusion
    {
        "id": "EN_ANX_01", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "Wait, I just received an automated margin call notice for eighteen million dollars due by 11 AM? I don't understand, our delta hedge was supposed to be completely neutral, how is this possible?!",
        "ser": "fear / anxiety", "conf": 0.95,
        "pitch": "210 Hz (rapid unstable jittery pitch leaps)", "tempo": "6.2 syll/sec (hyperventilating breathless tempo)", "energy": "72 dB (erratic shaky amplitude)", "vq": "severe vocal tremor, dry throat gulping"
    },
    {
        "id": "EN_ANX_02", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "I'm looking at the term sheet for the callable swaption barrier, and the Greeks don't match our models at all. Are we exposed to unlimited downside here? Please tell me we're not.",
        "ser": "confusion / panic", "conf": 0.91,
        "pitch": "195 Hz (wavering anxious pitch contour)", "tempo": "5.7 syll/sec (stumbling hesitation, re-starts)", "energy": "67 dB (shaky uneven volume)", "vq": "elevated shimmer 3.8%, nervous throat clears"
    },
    {
        "id": "EN_ANX_03", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "Our clearing broker is demanding additional tier-one equity by market close or they'll halt our trading privileges. What happens if our treasury wire gets delayed past cut-off?",
        "ser": "anxiety / worry", "conf": 0.93,
        "pitch": "205 Hz (high-pitched questioning intonation)", "tempo": "5.9 syll/sec (tense nervous pace)", "energy": "69 dB (strained fragile phonation)", "vq": "shallow hyperventilating breath sounds"
    },

    # 8. urgency_panic
    {
        "id": "EN_URG_01", "expected": "urgency_panic", "sarcastic": False,
        "text": "CRITICAL EMERGENCY! Our automated algorithmic market maker just went into an infinite loop and is submitting thousands of duplicate futures contracts! Kill the session NOW! PULL THE PLUG IMMEDIATELY!",
        "ser": "panic / emergency", "conf": 0.99,
        "pitch": "320 Hz (piercing emergency pitch scream)", "tempo": "7.8 syll/sec (frenzied machine-gun speed)", "energy": "92 dB (blaring high-intensity alarm tone)", "vq": "choked frantic airflow, vocal cord spasm"
    },
    {
        "id": "EN_URG_02", "expected": "urgency_panic", "sarcastic": False,
        "text": "We detected an unauthorized outbound wire transfer of forty-two million dollars right now! FREEZE ACCOUNT 992-ALPHA IMMEDIATELY, DO NOT LET THAT WIRE LEAVE CHIPS!",
        "ser": "urgency / panic", "conf": 0.98,
        "pitch": "305 Hz (commanding shriek, urgent attack)", "tempo": "7.3 syll/sec (extreme speech rate, zero pause)", "energy": "89 dB (explosive adrenaline saturation)", "vq": "hyper-tense laryngeal constriction"
    },
    {
        "id": "EN_URG_03", "expected": "urgency_panic", "sarcastic": False,
        "text": "The counterparty credit default rumor just flashed across the Bloomberg terminal and our gross exposure is eighty million! Cancel all open orders on that ticker this second!",
        "ser": "urgent panic", "conf": 0.97,
        "pitch": "290 Hz (sharp continuous pitch spikes)", "tempo": "7.0 syll/sec (rapid staccato alarm)", "energy": "87 dB (adrenaline surge, clipped breathing)", "vq": "rapid shallow gasps, urgent vocal drive"
    }
]

# -------------------------------------------------------------------------
# 24 JAPANESE-ONLY CALLS (3 CALLS PER EMOTION ACROSS ALL 8 EMOTIONS)
# -------------------------------------------------------------------------
JA_CALLS = [
    # 1. satisfaction_gratitude
    {
        "id": "JA_SAT_01", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "本日の円建てサムライ債の発行、想定以上の好条件で完売いただき大変感謝しております。貴行シンジケート団の迅速なご尽力に心より御礼申し上げます。",
        "ser": "gratitude / satisfaction (Kushinada HuBERT)", "conf": 0.97,
        "pitch": "155 Hz (温かみのある高揚したイントネーション)", "tempo": "6.8 mora/sec (落ち着いた丁寧なリズム)", "energy": "66 dB (明るく安定した胸声響き)", "vq": "低ジッター 0.3%, 柔らかな呼気"
    },
    {
        "id": "JA_SAT_02", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "クロスボーダーM&Aの買収資金調達が無事にクローズしました。非常にタイトなスケジュールの中、完璧な組成をしていただき安堵いたしました。",
        "ser": "joy / relief", "conf": 0.94,
        "pitch": "148 Hz (安堵感に満ちた穏やかなピッチ)", "tempo": "6.5 mora/sec (滑らかな発話テンポ)", "energy": "64 dB (安定した息遣い)", "vq": "声帯緊張なし, 自然な母音伸張"
    },
    {
        "id": "JA_SAT_03", "expected": "satisfaction_gratitude", "sarcastic": False,
        "text": "プライベート・エクイティ・ファンドの第1次募集、目標額を大幅に上回るコミットメントを確保できました。チームの皆様の素晴らしい成果に感謝いたします。",
        "ser": "satisfaction", "conf": 0.96,
        "pitch": "162 Hz (喜びと誇りを帯びた上昇調)", "tempo": "7.2 mora/sec (意欲的で活気のあるリズム)", "energy": "70 dB (力強く晴れやかな音量)", "vq": "明瞭な共鳴腔響き, 誠実な発声"
    },

    # 2. neutral_inquiry
    {
        "id": "JA_NEU_01", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "お世話になっております。先ほど約定いたしました長期国債先物の受渡決済期日と、担保受渡比率について確認させていただけますでしょうか。",
        "ser": "neutral", "conf": 0.98,
        "pitch": "125 Hz (平板で極めて落ち着いたトーン)", "tempo": "7.0 mora/sec (標準的なビジネス標準語速)", "energy": "61 dB (一定の平坦な音量)", "vq": "ジッター変動なし, 安定した発声"
    },
    {
        "id": "JA_NEU_02", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "当行のカストディ口座における外貨建てMMFの今月末の残高証明書の発行手続きについて、所要日数を教えてください。",
        "ser": "neutral", "conf": 0.97,
        "pitch": "120 Hz (一定の水平ピッチ輪郭)", "tempo": "6.9 mora/sec (正確で抑制された発話)", "energy": "60 dB (業務的で均一な音圧)", "vq": "正常なモード声, 感情起伏ゼロ"
    },
    {
        "id": "JA_NEU_03", "expected": "neutral_inquiry", "sarcastic": False,
        "text": "シンジケート・ローンの元利金返済スケジュール表の改訂版は、いつ頃ポータルサイトにアップロードされる予定でしょうか。",
        "ser": "neutral / calm", "conf": 0.96,
        "pitch": "122 Hz (事務的な平常ピッチ)", "tempo": "6.8 mora/sec (明瞭なアナウンス調)", "energy": "59 dB (淡々とした音量)", "vq": "声門閉鎖正常, 雑音なし"
    },

    # 3. frustration_irritation
    {
        "id": "JA_FRU_01", "expected": "frustration_irritation", "sarcastic": False,
        "text": "寄り付き直後から取引端末がフリーズして注文が通りません。これで今月に入って3回目ですよ。一体どういうインフラ管理をしているんですか？",
        "ser": "irritation / frustration", "conf": 0.90,
        "pitch": "172 Hz (語尾での鋭いピッチ急上昇)", "tempo": "8.5 mora/sec (早口で詰めるような語勢)", "energy": "73 dB (苛立ちを伴う強いアクセント)", "vq": "呼気圧上昇, 溜息の混入"
    },
    {
        "id": "JA_FRU_02", "expected": "frustration_irritation", "sarcastic": False,
        "text": "先週依頼した金利スワップの清算報告書がまだ届いていません。社内コンプライアンスの締め切りを過ぎているのですが、まだ時間がかかるのですか？",
        "ser": "annoyance", "conf": 0.89,
        "pitch": "165 Hz (語勢を強める圧迫ピッチ)", "tempo": "8.0 mora/sec (小刻みで不満げなリズム)", "energy": "71 dB (張り詰めた中音域)", "vq": "声帯の緊張, 苛立ちの舌打ち気味の音"
    },
    {
        "id": "JA_FRU_03", "expected": "frustration_irritation", "sarcastic": False,
        "text": "担当者が不在ということで何度も折り返しを待たされていますが、かれこれ4時間経っています。至急責任ある方から連絡をいただけませんか。",
        "ser": "frustration", "conf": 0.92,
        "pitch": "158 Hz (語尾が重く沈む不機嫌な音調)", "tempo": "7.8 mora/sec (不満を含んだ硬い調子)", "energy": "69 dB (押し殺した苛立ち)", "vq": "硬膜的発声, 声の硬直"
    },

    # 4. outrage_hostility
    {
        "id": "JA_OUT_01", "expected": "outrage_hostility", "sarcastic": False,
        "text": "事前通告もなしに勝手に担保を強制売却するとはどういう了見だ！明確な契約違反だぞ！即刻法務担当役員を出せ！金融庁にも告発してやる！",
        "ser": "anger / rage (Kushinada HuBERT)", "conf": 0.99,
        "pitch": "295 Hz (怒声による極度の高周波跳躍)", "tempo": "10.8 mora/sec (猛烈な怒気での連射発話)", "energy": "89 dB (マイクが歪む寸前の絶叫)", "vq": "ジッター 4.5%, 声帯の激しい乱れ"
    },
    {
        "id": "JA_OUT_02", "expected": "outrage_hostility", "sarcastic": False,
        "text": "貴社の誤発注のせいで当社は5億円の損失を被ったんだぞ！ふざけるな！損害賠償請求訴訟を今すぐ起こすから覚悟しておけ！",
        "ser": "outrage / fury", "conf": 0.98,
        "pitch": "280 Hz (激昂による破裂的ピッチ)", "tempo": "10.2 mora/sec (怒りを叩きつけるような怒号)", "energy": "87 dB (威嚇的な大音量)", "vq": "強い気流破裂音, 声の割れ"
    },
    {
        "id": "JA_OUT_03", "expected": "outrage_hostility", "sarcastic": False,
        "text": "ごまかすな！マーケットメイクを放棄して顧客を食い物にした詐欺行為だろうが！担当トレーダーの首を飛ばすまで絶対に納得しないからな！",
        "ser": "hostility / rage", "conf": 0.99,
        "pitch": "310 Hz (金切り声に近い怒りの極致)", "tempo": "11.1 mora/sec (息継ぎを忘れた激昂)", "energy": "92 dB (最大デシベルの怒声)", "vq": "喉頭の極度の収縮, 激しい震え"
    },

    # 5. disappointed_sad
    {
        "id": "JA_DIS_01", "expected": "disappointed_sad", "sarcastic": False,
        "text": "期待していた新規上場株式の主幹事公募割当、まさかゼロ株とは……あれだけ数ヶ月準備してきたプロジェクトが水の泡です。本当に失望しました。",
        "ser": "sadness / dejection", "conf": 0.93,
        "pitch": "82 Hz (地の底に落ちるような低ピッチ)", "tempo": "4.2 mora/sec (重々しく引きずる沈痛な速度)", "energy": "47 dB (消え入るようなかすれ声)", "vq": "高い気息性, 長い沈黙ポーズ"
    },
    {
        "id": "JA_DIS_02", "expected": "disappointed_sad", "sarcastic": False,
        "text": "劣後債の減損処理が確定してしまったのですね。創業から支えてきた企業だっただけに、このような結末になり言葉もありません……。",
        "ser": "grief / sorrow", "conf": 0.95,
        "pitch": "85 Hz (失意に沈む下垂ピッチ)", "tempo": "4.5 mora/sec (力なく途切れ途切れの発話)", "energy": "49 dB (弱々しい沈黙混じりの声)", "vq": "声の震え, 深い落胆のため息"
    },
    {
        "id": "JA_DIS_03", "expected": "disappointed_sad", "sarcastic": False,
        "text": "当ファンドの月次運用レポートを拝見しました。マイナス30パーセントのドローダウンですか……もう投資家に対して顔向けができません。",
        "ser": "hopelessness / sadness", "conf": 0.92,
        "pitch": "80 Hz (抑揚のない絶望的な単調音)", "tempo": "3.9 mora/sec (気力を失った遅延テンポ)", "energy": "45 dB (虚ろな掠れ声)", "vq": "高周波帯域の欠落, 諦念の息"
    },

    # 6. sarcasm_passive_aggression
    {
        "id": "JA_SAR_01", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "いやあ、実に素晴らしいマーケットメイク能力ですね。指値から50銭も不利な価格で約定させてくださるとは、さすが超一流の投資銀行様ですこと。",
        "ser": "anger / hostility (音響的韻律特徴)", "conf": 0.92,
        "pitch": "235 Hz (芝居がかった過剰な抑揚の上昇)", "tempo": "5.2 mora/sec (引き延ばされた冷笑的母音)", "energy": "76 dB (嫌味を含んだ刺々しい強勢)", "vq": "皮肉な緊張音, 冷淡な嘲笑トーン"
    },
    {
        "id": "JA_SAR_02", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "急ぎの担保差し替え申請に丸4日もかけていただき、まことに手厚いご対応感服いたしました。おかげで市場機会を完全に逃せましたよ、素晴らしいお仕事です。",
        "ser": "contempt / sarcasm", "conf": 0.95,
        "pitch": "220 Hz (慇懃無礼な作為的高音)", "tempo": "5.5 mora/sec (嫌味たらしい丁寧語の強調)", "energy": "74 dB (抑圧された激しい怒り)", "vq": "冷たい舌打ち気味の音節, 偽りの敬語"
    },
    {
        "id": "JA_SAR_03", "expected": "sarcasm_passive_aggression", "sarcastic": True,
        "text": "日銀金融政策決定会合の直前というこの上ないタイミングで回線を遮断してくださり、本当に最高のサポート体制ですね。敬服いたします。",
        "ser": "passive aggressive / anger", "conf": 0.94,
        "pitch": "228 Hz (誇張された称賛風のピッチ山)", "tempo": "5.3 mora/sec (皮肉を込めたゆったりした節回し)", "energy": "75 dB (牙を隠した攻撃的音圧)", "vq": "声門の鋭い収縮, 敵意を秘めた含み笑い"
    },

    # 7. anxiety_confusion
    {
        "id": "JA_ANX_01", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "ちょっと待ってください！今朝の追加証拠金請求が20億円って本当ですか！？完全デルタニュートラルにヘッジしていたはずなのに何故こんな金額になるんですか！？",
        "ser": "panic / anxiety (Kushinada HuBERT)", "conf": 0.96,
        "pitch": "215 Hz (動揺による高周波の不安定な跳ね上がり)", "tempo": "9.5 mora/sec (過呼吸気味の切迫したテンポ)", "energy": "74 dB (震えるような不安定な音量)", "vq": "激しい声の震え, 唾を呑み込む音"
    },
    {
        "id": "JA_ANX_02", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "スワップションのタームシートを見ているのですが、この条項だと金利急騰時に損失が青天井になりませんか？私の理解が間違っているのでしょうか、不安で仕方がありません……。",
        "ser": "confusion / worry", "conf": 0.92,
        "pitch": "198 Hz (語尾が震えて上ずるピッチ輪郭)", "tempo": "8.8 mora/sec (言い直しや言い淀みの多発)", "energy": "68 dB (頼りなく揺れる声量)", "vq": "シマー 3.9%, 不安げな息切れ"
    },
    {
        "id": "JA_ANX_03", "expected": "anxiety_confusion", "sarcastic": False,
        "text": "大引けまでに清算基金への追証が着金しないと口座凍結されると言われました。銀行間の送金遅延が起きたらどうなるんですか？当社は破綻してしまうんですか……！？",
        "ser": "fear / anxiety", "conf": 0.94,
        "pitch": "208 Hz (切迫した悲鳴に近い上ずり)", "tempo": "9.2 mora/sec (切迫した緊張した間隔)", "energy": "70 dB (必死に救いを求めるトーン)", "vq": "浅い過呼吸音, 緊迫した喉の締め付け"
    },

    # 8. urgency_panic
    {
        "id": "JA_URG_01", "expected": "urgency_panic", "sarcastic": False,
        "text": "緊急事態です！当社の自動取引アルゴリズムが暴走して先物の買い注文を無限ループで出し続けています！今すぐセッションを切断してください！全注文を今すぐ強制キャンセルしろ！！",
        "ser": "emergency / panic (Kushinada HuBERT)", "conf": 0.99,
        "pitch": "330 Hz (突き刺さるような切迫絶叫)", "tempo": "12.5 mora/sec (超音速マシンガン発話)", "energy": "93 dB (極限のパニック警報音圧)", "vq": "息も絶え絶えの切迫感, 破裂的発声"
    },
    {
        "id": "JA_URG_02", "expected": "urgency_panic", "sarcastic": False,
        "text": "不正アクセスによる50億円の海外不正送金が現在進行形で発生しています！口座番号883の出金処理を今すぐ全停止してください！一秒も待てません！！",
        "ser": "urgency / panic", "conf": 0.98,
        "pitch": "315 Hz (非常ベルのような緊急ピッチ)", "tempo": "11.8 mora/sec (間髪入れない緊急命令口調)", "energy": "90 dB (切迫した大音量)", "vq": "咽頭の極度の緊張, 緊迫した呼気"
    },
    {
        "id": "JA_URG_03", "expected": "urgency_panic", "sarcastic": False,
        "text": "取引先メガバンクの破綻報道がブルームバーグ速報で出ました！当社の信用エクスポージャーが100億あります！関連する全オープンポジションを直ちに市場で外せ！！",
        "ser": "urgent panic", "conf": 0.97,
        "pitch": "298 Hz (緊張で張り詰めた金属音調)", "tempo": "11.2 mora/sec (緊迫した連射調)", "energy": "88 dB (アドレナリン全開の緊迫声)", "vq": "急速な浅い息継ぎ, 切迫した推進力"
    }
]

print("=" * 80)
print(f"RUNNING 24 ENGLISH-ONLY CALLS...")
print("=" * 80)
print(f"{'ID':<11} | {'Expected Emotion':<26} | {'Predicted Emotion':<26} | {'Conf':<6} | {'Sarc(P)':<7} | {'Lat(ms)':<7} | {'Match'}")
print("-" * 98)

en_results = []
en_latencies = []

for c in EN_CALLS:
    prompt = format_prompt(
        transcript=c["text"],
        ser_prior=c["ser"],
        ser_conf=c["conf"],
        pitch_cue=c["pitch"],
        tempo_cue=c["tempo"],
        energy_cue=c["energy"],
        voice_quality=c["vq"]
    )
    t_start = time.perf_counter()
    with torch.inference_mode():
        res = router.predict(prompt, IB_QUESTIONS, model="multilingual")
    lat_ms = (time.perf_counter() - t_start) * 1000
    en_latencies.append(lat_ms)
    
    pred_emo = res["answers"]["detected_emotion"]["choice"]
    pred_conf = res["answers"]["detected_emotion"]["answer_confidence"]
    sarc_p = res["answers"]["sarcasm_detected"]["noul"]
    match = (pred_emo == c["expected"])
    
    status_str = "MATCH [PASS]" if match else "MISMATCH"
    print(f"{c['id']:<11} | {c['expected']:<26} | {pred_emo:<26} | {pred_conf:>5.1%} | {sarc_p:>5.1%} | {lat_ms:>6.1f} | {status_str}")
    
    en_results.append({
        "id": c["id"],
        "lang": "en",
        "expected": c["expected"],
        "predicted": pred_emo,
        "confidence": pred_conf,
        "sarcasm_p": sarc_p,
        "match": match,
        "latency_ms": lat_ms
    })
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

en_acc = sum(1 for r in en_results if r["match"]) / len(en_results)
en_avg_lat = sum(en_latencies) / len(en_latencies)
print(f"\n[+] English 24-Call Accuracy : {en_acc:.1%} ({sum(1 for r in en_results if r['match'])}/24)")
print(f"[+] English Avg Latency     : {en_avg_lat:.1f} ms\n")

print("=" * 80)
print(f"RUNNING 24 JAPANESE-ONLY CALLS...")
print("=" * 80)
print(f"{'ID':<11} | {'Expected Emotion':<26} | {'Predicted Emotion':<26} | {'Conf':<6} | {'Sarc(P)':<7} | {'Lat(ms)':<7} | {'Match'}")
print("-" * 98)

ja_results = []
ja_latencies = []

for c in JA_CALLS:
    prompt = format_prompt(
        transcript=c["text"],
        ser_prior=c["ser"],
        ser_conf=c["conf"],
        pitch_cue=c["pitch"],
        tempo_cue=c["tempo"],
        energy_cue=c["energy"],
        voice_quality=c["vq"]
    )
    t_start = time.perf_counter()
    with torch.inference_mode():
        res = router.predict(prompt, IB_QUESTIONS, model="multilingual")
    lat_ms = (time.perf_counter() - t_start) * 1000
    ja_latencies.append(lat_ms)
    
    pred_emo = res["answers"]["detected_emotion"]["choice"]
    pred_conf = res["answers"]["detected_emotion"]["answer_confidence"]
    sarc_p = res["answers"]["sarcasm_detected"]["noul"]
    match = (pred_emo == c["expected"])
    
    status_str = "MATCH [PASS]" if match else "MISMATCH"
    print(f"{c['id']:<11} | {c['expected']:<26} | {pred_emo:<26} | {pred_conf:>5.1%} | {sarc_p:>5.1%} | {lat_ms:>6.1f} | {status_str}")
    
    ja_results.append({
        "id": c["id"],
        "lang": "ja",
        "expected": c["expected"],
        "predicted": pred_emo,
        "confidence": pred_conf,
        "sarcasm_p": sarc_p,
        "match": match,
        "latency_ms": lat_ms
    })
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

ja_acc = sum(1 for r in ja_results if r["match"]) / len(ja_results)
ja_avg_lat = sum(ja_latencies) / len(ja_latencies)
print(f"\n[+] Japanese 24-Call Accuracy : {ja_acc:.1%} ({sum(1 for r in ja_results if r['match'])}/24)")
print(f"[+] Japanese Avg Latency     : {ja_avg_lat:.1f} ms\n")

# Combined Metrics
total_results = en_results + ja_results
total_acc = sum(1 for r in total_results if r["match"]) / len(total_results)
total_lat = sum(en_latencies + ja_latencies) / len(total_results)

# Sarcasm Metrics
sarcastic_calls = [r for r in total_results if "SAR" in r["id"]]
sarc_detected = sum(1 for r in sarcastic_calls if r["sarcasm_p"] > 0.5)
non_sarcastic_calls = [r for r in total_results if "SAR" not in r["id"]]
false_sarc = sum(1 for r in non_sarcastic_calls if r["sarcasm_p"] > 0.5)

print("=" * 80)
print("FINAL PIPELINE EVALUATION SUMMARY (48 CALLS)")
print("=" * 80)
print(f"Overall Accuracy Across 8 Emotions : {total_acc:.1%} ({sum(1 for r in total_results if r['match'])}/48)")
print(f"English-Only Accuracy (24 Calls)   : {en_acc:.1%} ({sum(1 for r in en_results if r['match'])}/24)")
print(f"Japanese-Only Accuracy (24 Calls)  : {ja_acc:.1%} ({sum(1 for r in ja_results if r['match'])}/24)")
print(f"Average Inference Latency (CUDA)   : {total_lat:.1f} ms")
print(f"Sarcasm True Positive Rate (Sensitivity): {sarc_detected}/{len(sarcastic_calls)} ({sarc_detected/len(sarcastic_calls):.1%})")
print(f"Sarcasm False Alarm Rate (Specificity) : {false_sarc}/{len(non_sarcastic_calls)} ({false_sarc/len(non_sarcastic_calls):.1%})")
print("=" * 80)

# Save to json
summary_data = {
    "total_calls": 48,
    "overall_accuracy": total_acc,
    "english_accuracy": en_acc,
    "japanese_accuracy": ja_acc,
    "average_latency_ms": total_lat,
    "sarcasm_recall": sarc_detected / len(sarcastic_calls),
    "sarcasm_false_positive_rate": false_sarc / len(non_sarcastic_calls),
    "english_results": en_results,
    "japanese_results": ja_results
}

with open("benchmark_multilingual_24x2_results.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)
print("Saved complete benchmark results to benchmark_multilingual_24x2_results.json")
