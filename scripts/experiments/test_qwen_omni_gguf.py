import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def run_qwen_omni(wav_path: str, prompt: str, title: str):
    print("=" * 84)
    print(f"TEST: {title}")
    print(f"Audio: {wav_path}")
    print("=" * 84)
    t0 = time.perf_counter()
    cmd = [
        str(ROOT / "tools" / "llama" / "llama-mtmd-cli.exe"),
        "-m", str(ROOT / "data" / "models" / "Qwen2.5-Omni-3B-Q4_K_M.gguf"),
        "--mmproj", str(ROOT / "data" / "models" / "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf"),
        "--audio", str(ROOT / wav_path),
        "-c", "2048",
        "-ngl", "28",
        "--temp", "0.0",
        "-n", "220",
        "-p", prompt,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - t0
    print(f"[Elapsed: {elapsed:.2f}s]")
    print("OUTPUT:\n" + res.stdout.strip() + "\n")

def main():
    # Mode 1: Pure Native Audio (Zero Text Hint)
    p1 = (
        "<__media__>\n"
        "Listen to the speaker's voice in this audio clip. "
        "Describe their vocal emotion, tone of voice, and whether they sound genuinely happy, sad, or sarcastic."
    )
    run_qwen_omni("scratch/rec_1791807581_16k_norm.wav", p1, "Qwen2.5-Omni-3B-Q4_K_M (Pure Audio - Zero Text Prior)")

    # Mode 2: Audio + Faster-Whisper Transcript
    p2 = (
        "<__media__>\n"
        "Spoken words (from ASR): \"Thank you so much for the help today.\"\n"
        "Listen carefully to the speaker's actual vocal tone, pitch, and energy in the audio above. "
        "Notice any contradiction between the polite words and the vocal tone. "
        "Classify into one of the 8 canonical emotions: "
        "[satisfaction_gratitude, neutral_inquiry, frustration_irritation, outrage_hostility, "
        "disappointed_sad, sarcasm_passive_aggression, anxiety_confusion, urgency_panic] "
        "and 3-way sentiment [positive, neutral, negative]. "
        "Respond strictly in JSON with keys: detected_emotion, confidence, sarcasm_detected, sentiment, reasoning."
    )
    run_qwen_omni("scratch/rec_1791807581_16k_norm.wav", p2, "Qwen2.5-Omni-3B-Q4_K_M (Audio + Whisper Transcript)")

if __name__ == "__main__":
    main()
