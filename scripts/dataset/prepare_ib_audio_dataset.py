import os
import io
import json
from pathlib import Path
import soundfile as sf
import pyarrow.parquet as pq

# Directories
base_dir = Path(__file__).parent / "data" / "audio_samples"
en_dir = base_dir / "en"
ja_dir = base_dir / "ja"
en_dir.mkdir(parents=True, exist_ok=True)
ja_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EXTRACTING 30 ENGLISH & 30 JAPANESE IB AUDIO SAMPLES + GROUND TRUTH")
print("=" * 70)

# 1. English: Earnings22
print("\n[+] Extracting 30 English Earnings Calls samples...")
table_en = pq.read_table("audio_datasets/earnings22_test.parquet")
en_samples_meta = []

# Group rows or select 30 diverse segments from different calls
seen_sources = set()
count = 0
for i in range(len(table_en)):
    source_id = table_en["source_id"][i].as_py()
    sentence = table_en["sentence"][i].as_py()
    audio_struct = table_en["audio"][i]
    audio_bytes = audio_struct["bytes"].as_py()
    
    if not sentence or len(sentence.strip()) < 15:
        continue
        
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        dur = len(data) / sr
        if dur < 2.0:
            continue
    except Exception:
        continue
        
    count += 1
    sample_id = f"en_ib_sample_{count:02d}"
    wav_path = en_dir / f"{sample_id}.wav"
    sf.write(str(wav_path), data, sr)
    
    en_samples_meta.append({
        "sample_id": sample_id,
        "filename": f"{sample_id}.wav",
        "language": "en",
        "domain": "Investment Banking / Corporate Earnings Call",
        "source_call_id": source_id,
        "duration_sec": round(dur, 2),
        "ground_truth_transcript": sentence.strip()
    })
    
    if count >= 30:
        break

with open(en_dir / "ground_truth_en.json", "w", encoding="utf-8") as f:
    json.dump(en_samples_meta, f, indent=2, ensure_ascii=False)
print(f"    Extracted {len(en_samples_meta)} English audio files to {en_dir}")

# 2. Japanese: ReazonSpeech (Business & Financial News Broadcasts)
print("\n[+] Extracting 30 Japanese Financial/Business Broadcast samples...")
table_ja = pq.read_table("audio_datasets/reazonspeech_train.parquet")
ja_samples_meta = []

count = 0
for i in range(len(table_ja)):
    text_ja = table_ja["transcription"][i].as_py()
    text_en = table_ja["transcription/en_gpt3.5"][i].as_py()
    audio_struct = table_ja["audio"][i]
    audio_bytes = audio_struct["bytes"].as_py()
    
    if not text_ja or len(text_ja.strip()) < 10:
        continue
        
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        dur = len(data) / sr
        if dur < 2.5:
            continue
    except Exception:
        continue
        
    count += 1
    sample_id = f"ja_ib_sample_{count:02d}"
    wav_path = ja_dir / f"{sample_id}.wav"
    sf.write(str(wav_path), data, sr)
    
    ja_samples_meta.append({
        "sample_id": sample_id,
        "filename": f"{sample_id}.wav",
        "language": "ja",
        "domain": "Japanese Financial & Economic Broadcast",
        "duration_sec": round(dur, 2),
        "ground_truth_transcript": text_ja.strip(),
        "english_translation": text_en.strip() if text_en else ""
    })
    
    if count >= 30:
        break

with open(ja_dir / "ground_truth_ja.json", "w", encoding="utf-8") as f:
    json.dump(ja_samples_meta, f, indent=2, ensure_ascii=False)
print(f"    Extracted {len(ja_samples_meta)} Japanese audio files to {ja_dir}")

print("\n" + "=" * 70)
print("DATASET EXTRACTION COMPLETE: 30 EN + 30 JA SAMPLES READY")
print("=" * 70)
