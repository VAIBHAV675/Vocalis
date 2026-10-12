import json
from pathlib import Path

base_dir = Path("data/audio_samples")
en_dir = base_dir / "en"
ja_dir = base_dir / "ja"

# 1. English
with open(en_dir / "ground_truth_en.json", "r", encoding="utf-8") as f:
    en_data = json.load(f)

en_md_lines = [
    "# English Investment Banking Audio Samples - Ground Truth Transcripts",
    "",
    "Source Dataset: Earnings22 (`sanchit-gandhi/earnings22_robust_split`)",
    "Domain: Corporate Earnings Calls & Investment Banking Presentations",
    "",
    "| Sample ID | File | Duration | Ground Truth Transcript |",
    "| :--- | :--- | :---: | :--- |"
]

for item in en_data:
    txt_path = en_dir / f"{item['sample_id']}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Sample: {item['filename']}\n")
        f.write(f"Duration: {item['duration_sec']}s\n")
        f.write(f"Call ID: {item.get('source_call_id', 'N/A')}\n")
        f.write("-" * 50 + "\n")
        f.write(item["ground_truth_transcript"] + "\n")
    
    en_md_lines.append(
        f"| **{item['sample_id']}** | `{item['filename']}` | {item['duration_sec']}s | {item['ground_truth_transcript']} |"
    )

with open(en_dir / "HUMAN_READABLE_TRANSCRIPTS.md", "w", encoding="utf-8") as f:
    f.write("\n".join(en_md_lines) + "\n")

print(f"Created 30 .txt files and HUMAN_READABLE_TRANSCRIPTS.md in {en_dir}")

# 2. Japanese
with open(ja_dir / "ground_truth_ja.json", "r", encoding="utf-8") as f:
    ja_data = json.load(f)

ja_md_lines = [
    "# Japanese Financial & Business Audio Samples - Ground Truth Transcripts",
    "",
    "Source Dataset: ReazonSpeech (`japanese-asr/ja_asr.reazon_speech_all`)",
    "Domain: Japanese Financial & Economic Broadcasts",
    "",
    "| Sample ID | File | Duration | Japanese Ground Truth | English Translation |",
    "| :--- | :--- | :---: | :--- | :--- |"
]

for item in ja_data:
    txt_path = ja_dir / f"{item['sample_id']}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Sample: {item['filename']}\n")
        f.write(f"Duration: {item['duration_sec']}s\n")
        f.write("-" * 50 + "\n")
        f.write(f"Japanese: {item['ground_truth_transcript']}\n")
        if item.get("english_translation"):
            f.write(f"English:  {item['english_translation']}\n")
            
    ja_md_lines.append(
        f"| **{item['sample_id']}** | `{item['filename']}` | {item['duration_sec']}s | {item['ground_truth_transcript']} | {item.get('english_translation', '')} |"
    )

with open(ja_dir / "HUMAN_READABLE_TRANSCRIPTS.md", "w", encoding="utf-8") as f:
    f.write("\n".join(ja_md_lines) + "\n")

print(f"Created 30 .txt files and HUMAN_READABLE_TRANSCRIPTS.md in {ja_dir}")
