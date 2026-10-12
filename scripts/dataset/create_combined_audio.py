import os
import json
import numpy as np
import soundfile as sf
import librosa

def combine_files(file_list, output_wav, metadata_json, lang, pause_sec=0.5, target_sr=16000):
    pause_samples = int(pause_sec * target_sr)
    combined_audio = []
    timeline = []
    current_sample = 0

    print(f"\n[+] Combining {len(file_list)} files for {lang.upper()} into {output_wav} (Target SR: {target_sr} Hz)...")

    for idx, fpath in enumerate(file_list, 1):
        data, file_sr = sf.read(fpath)
        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # PROPER HIGH-QUALITY RESAMPLING
        if file_sr != target_sr:
            print(f"  Resampling clip #{idx:02d} ({os.path.basename(fpath)}): {file_sr} Hz -> {target_sr} Hz")
            data = librosa.resample(data, orig_sr=file_sr, target_sr=target_sr, res_type="soxr_hq")

        # Normalize volume to avoid clipping
        max_val = np.max(np.abs(data))
        if max_val > 0.01:
            data = (data / max_val) * 0.95

        start_time = current_sample / target_sr
        current_sample += len(data)
        end_time = current_sample / target_sr

        combined_audio.append(data)
        timeline.append({
            "index": idx,
            "filename": os.path.basename(fpath),
            "start_sec": round(start_time, 2),
            "end_sec": round(end_time, 2),
            "duration_sec": round(end_time - start_time, 2)
        })

        # Add pause between samples
        if idx < len(file_list):
            combined_audio.append(np.zeros(pause_samples, dtype=np.float32))
            current_sample += pause_samples

    final_waveform = np.concatenate(combined_audio)
    sf.write(output_wav, final_waveform, target_sr, subtype="PCM_16")
    total_duration = len(final_waveform) / target_sr

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump({
            "language": lang,
            "total_files": len(file_list),
            "target_samplerate": target_sr,
            "total_duration_sec": round(total_duration, 2),
            "timeline": timeline
        }, f, indent=2)

    print(f"[✓] Successfully combined {len(file_list)} files into {output_wav} ({total_duration:.1f}s)")
    return output_wav

def main():
    en_files = [f"data/audio_samples/en/en_ib_sample_{i:02d}.wav" for i in range(1, 31)]
    ja_files = [f"data/audio_samples/ja/ja_ib_sample_{i:02d}.wav" for i in range(1, 31)]

    combine_files(en_files, "data/audio_samples/combined_english_30_calls.wav", "data/audio_samples/combined_english_timeline.json", "en")
    combine_files(ja_files, "data/audio_samples/combined_japanese_30_broadcasts.wav", "data/audio_samples/combined_japanese_timeline.json", "ja")

if __name__ == "__main__":
    main()
