import librosa
import numpy as np
import soundfile as sf

def extract_prosody_features(wav_path: str) -> dict:
    """
    Extracts self-contained prosodic and acoustic cues using librosa.
    Returns human-interpretable and LLM-ready descriptions.
    """
    y, sr = librosa.load(wav_path, sr=16000)
    duration = len(y) / sr
    
    # 1. Pitch (F0) using YIN
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    valid_f0 = f0[(f0 > 50) & (f0 < 400)]
    if len(valid_f0) > 0:
        mean_pitch = float(np.mean(valid_f0))
        std_pitch = float(np.std(valid_f0))
        # Contour description
        if std_pitch < 15:
            contour_desc = "flat monotonic contour"
        elif np.mean(valid_f0[-len(valid_f0)//4:]) > np.mean(valid_f0[:len(valid_f0)//4]) + 20:
            contour_desc = "rising intonation / questioning / tension"
        elif np.mean(valid_f0[-len(valid_f0)//4:]) < np.mean(valid_f0[:len(valid_f0)//4]) - 20:
            contour_desc = "downward falling terminal contour"
        else:
            contour_desc = "natural melodic modulation"
        pitch_str = f"{mean_pitch:.1f} Hz (std: {std_pitch:.1f} Hz, {contour_desc})"
    else:
        pitch_str = "unvoiced / whispering / low energy"
    
    # 2. Energy (RMS dB)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    # dB calculation relative to full scale
    rms_db = float(20 * np.log10(mean_rms + 1e-6) + 100) # normalized ~30-90 dB scale
    if rms_db > 78:
        energy_desc = f"{rms_db:.1f} dB (high intensity / elevated volume)"
    elif rms_db < 55:
        energy_desc = f"{rms_db:.1f} dB (subdued / quiet / low intensity)"
    else:
        energy_desc = f"{rms_db:.1f} dB (moderate baseline business volume)"
        
    # 3. Tempo (Onset strength / speaking rate)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    syll_rate = len(onsets) / max(duration, 0.5)
    if syll_rate > 6.0:
        tempo_desc = f"{syll_rate:.1f} onsets/sec (fast / rushed / tense speaking rate)"
    elif syll_rate < 3.0:
        tempo_desc = f"{syll_rate:.1f} onsets/sec (slow / measured / hesitant pace)"
    else:
        tempo_desc = f"{syll_rate:.1f} onsets/sec (steady comfortable conversational tempo)"
        
    # 4. Voice Quality (Spectral Centroid & Flatness)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    
    if flatness > 0.05:
        vq_desc = "breathy / aspirated phonation"
    elif centroid > 2500:
        vq_desc = "sharp / bright / tense acoustic resonance"
    elif centroid < 1200:
        vq_desc = "deep / dark / chest resonance"
    else:
        vq_desc = "standard modal phonation, clean harmonic structure"
        
    return {
        "duration_sec": duration,
        "pitch_str": pitch_str,
        "energy_str": energy_desc,
        "tempo_str": tempo_desc,
        "voice_quality_str": vq_desc,
        "raw_features": {
            "mean_f0": mean_pitch if len(valid_f0) > 0 else 0.0,
            "std_f0": std_pitch if len(valid_f0) > 0 else 0.0,
            "rms_db": rms_db,
            "syll_rate": syll_rate,
            "centroid": centroid,
            "flatness": flatness
        }
    }

if __name__ == "__main__":
    res = extract_prosody_features("data/audio_samples/en/en_ib_sample_01.wav")
    print("Test Prosody Extraction on en_ib_sample_01:")
    for k, v in res.items():
        if k != "raw_features":
            print(f"  {k}: {v}")
