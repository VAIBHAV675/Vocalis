import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
base_dir = ROOT / "data" / "audio_samples"
en_dir = base_dir / "en"
ja_dir = base_dir / "ja"

def main():
    master_path = base_dir / "ground_truth_2000_cleaned.json"
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    en_data = master.get("english_samples", [])
    ja_data = master.get("japanese_samples", [])

    # 1. English (1,000 files)
    en_md_lines = [
        "# English Investment Banking Audio Samples - Ground Truth Transcripts (1,000 Clips)",
        "",
        "Source Dataset: Earnings22 (`sanchit-gandhi/earnings22_robust_split`)",
        "Domain: Corporate Earnings Calls & Investment Banking Presentations",
        "",
        "| Sample ID | File | Duration | Emotion | Sentiment | Ground Truth Transcript |",
        "| :--- | :--- | :---: | :--- | :---: | :--- |"
    ]

    en_count = 0
    for item in en_data:
        txt_path = en_dir / f"{item['sample_id']}.txt"
        emo = item.get("ground_truth_emotion", "neutral_inquiry")
        sent = item.get("ground_truth_sentiment", "neutral")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Sample: {item['filename']}\n")
            f.write(f"Duration: {item['duration_sec']}s\n")
            f.write(f"Call ID: {item.get('source_call_id', 'N/A')}\n")
            f.write(f"Ground Truth Emotion: {emo}\n")
            f.write(f"Ground Truth Sentiment: {sent}\n")
            f.write("-" * 50 + "\n")
            f.write(item["ground_truth_transcript"] + "\n")
        en_count += 1

        en_md_lines.append(
            f"| **{item['sample_id']}** | `{item['filename']}` | {item['duration_sec']}s | `{emo}` | `{sent}` | {item['ground_truth_transcript']} |"
        )

    with open(en_dir / "HUMAN_READABLE_TRANSCRIPTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(en_md_lines) + "\n")

    print(f"[✓] Created {en_count} .txt files and HUMAN_READABLE_TRANSCRIPTS.md in {en_dir}")

    # 2. Japanese (1,000 files)
    ja_md_lines = [
        "# Japanese Financial & Business Audio Samples - Ground Truth Transcripts (1,000 Clips)",
        "",
        "Source Dataset: ReazonSpeech (`japanese-asr/ja_asr.reazon_speech_all`)",
        "Domain: Japanese Financial & Economic Broadcasts",
        "",
        "| Sample ID | File | Duration | Emotion | Sentiment | Japanese Ground Truth | English Translation |",
        "| :--- | :--- | :---: | :--- | :---: | :--- | :--- |"
    ]

    ja_count = 0
    for item in ja_data:
        txt_path = ja_dir / f"{item['sample_id']}.txt"
        emo = item.get("ground_truth_emotion", "neutral_inquiry")
        sent = item.get("ground_truth_sentiment", "neutral")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Sample: {item['filename']}\n")
            f.write(f"Duration: {item['duration_sec']}s\n")
            f.write(f"Ground Truth Emotion: {emo}\n")
            f.write(f"Ground Truth Sentiment: {sent}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Japanese: {item['ground_truth_transcript']}\n")
            if item.get("english_translation"):
                f.write(f"English:  {item['english_translation']}\n")
        ja_count += 1

        ja_md_lines.append(
            f"| **{item['sample_id']}** | `{item['filename']}` | {item['duration_sec']}s | `{emo}` | `{sent}` | {item['ground_truth_transcript']} | {item.get('english_translation', '')} |"
        )

    with open(ja_dir / "HUMAN_READABLE_TRANSCRIPTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(ja_md_lines) + "\n")

    print(f"[✓] Created {ja_count} .txt files and HUMAN_READABLE_TRANSCRIPTS.md in {ja_dir}")

if __name__ == "__main__":
    main()
