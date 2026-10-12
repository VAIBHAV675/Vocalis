import os
import io
import json
import time
from pathlib import Path
import soundfile as sf
import pyarrow.parquet as pq

def main():
    t0 = time.time()
    base_dir = Path(__file__).parent / "data" / "audio_samples"
    en_dir = base_dir / "en"
    ja_dir = base_dir / "ja"
    en_dir.mkdir(parents=True, exist_ok=True)
    ja_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTRACTING 1,000 ENGLISH & 1,000 JAPANESE AUDIO SAMPLES (2,000 TOTAL)")
    print("=" * 80)

    # 1. English: Earnings22 (Shard 1 + Shard 2)
    print("\n[+] Extracting 1,000 English Earnings22 samples...")
    en_shards = [
        "audio_datasets/earnings22_test.parquet",
        "audio_datasets/earnings22_test_shard2.parquet"
    ]
    en_samples_meta = []
    count = 0

    for shard_path in en_shards:
        if not os.path.exists(shard_path):
            print(f"    [!] Warning: {shard_path} not found, skipping.")
            continue
        table_en = pq.read_table(shard_path)
        print(f"    Loaded {shard_path} ({len(table_en)} rows)")
        cols = set(table_en.column_names)
        src_col = "source_id" if "source_id" in cols else "file_id"
        txt_col = "sentence" if "sentence" in cols else "transcription"
        for i in range(len(table_en)):
            source_id = table_en[src_col][i].as_py()
            sentence = table_en[txt_col][i].as_py()
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
            if not wav_path.exists():
                sf.write(str(wav_path), data, sr)

            en_samples_meta.append({
                "sample_id": sample_id,
                "filename": f"{sample_id}.wav",
                "wav_path": f"data/audio_samples/en/{sample_id}.wav",
                "language": "en",
                "domain": "Investment Banking / Corporate Earnings Call",
                "source_call_id": str(source_id),
                "duration_sec": round(dur, 2),
                "ground_truth_transcript": sentence.strip()
            })

            if count >= 1000:
                break
        if count >= 1000:
            break

    with open(en_dir / "ground_truth_en_1000.json", "w", encoding="utf-8") as f:
        json.dump(en_samples_meta, f, indent=2, ensure_ascii=False)
    print(f"    [✓] Extracted {len(en_samples_meta)} English audio files to {en_dir}")

    # 2. Japanese: ReazonSpeech
    print("\n[+] Extracting 1,000 Japanese ReazonSpeech Broadcast samples...")
    table_ja = pq.read_table("audio_datasets/reazonspeech_train.parquet")
    print(f"    Loaded audio_datasets/reazonspeech_train.parquet ({len(table_ja)} rows)")
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
        if not wav_path.exists():
            sf.write(str(wav_path), data, sr)

        ja_samples_meta.append({
            "sample_id": sample_id,
            "filename": f"{sample_id}.wav",
            "wav_path": f"data/audio_samples/ja/{sample_id}.wav",
            "language": "ja",
            "domain": "Japanese Financial & Economic Broadcast",
            "duration_sec": round(dur, 2),
            "ground_truth_transcript": text_ja.strip(),
            "english_translation": text_en.strip() if text_en else ""
        })

        if count >= 1000:
            break

    with open(ja_dir / "ground_truth_ja_1000.json", "w", encoding="utf-8") as f:
        json.dump(ja_samples_meta, f, indent=2, ensure_ascii=False)
    print(f"    [✓] Extracted {len(ja_samples_meta)} Japanese audio files to {ja_dir}")

    print("\n" + "=" * 80)
    print(f"DATASET EXTRACTION COMPLETE IN {time.time() - t0:.2f}s: {len(en_samples_meta)} EN + {len(ja_samples_meta)} JA")
    print("=" * 80)

if __name__ == "__main__":
    main()
