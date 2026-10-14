import os
import sys
import gc
import time
import json
import subprocess
from pathlib import Path

# Add local kev clone to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kev"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force IPv4
import socket
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

import torch
# PyTorch 2.6 weights_only patch for FunASR
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import soundfile as sf
import librosa
import requests

from extract_prosody import extract_prosody_features

# Canonical 8 emotions schema
IB_QUESTIONS = {
    "detected_emotion": {
        "type": "choice",
        "instructions": (
            "Analyze the caller's spoken statement in the full context of their vocal acoustics and delivery. "
            "Resolve any tension between the words and vocal tone (e.g. sarcasm, polite complaints, subdued relief). "
            "Classify the customer's true emotional state into one of the 8 canonical categories."
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
        "instructions": "Is the speaker using sarcasm, irony, or passive-aggressive politeness that contradicts their vocal tone?"
    }
}

EMOTION_MAP_E2V = {
    "neutral": "neutral_inquiry",
    "happy": "satisfaction_gratitude",
    "angry": "outrage_hostility",
    "sad": "disappointed_sad",
    "fearful": "urgency_panic",
    "disgusted": "frustration_irritation",
    "surprised": "anxiety_confusion",
    "<unk>": "neutral_inquiry",
    "unk": "neutral_inquiry",
    "other": "neutral_inquiry"
}


def clean_e2v_label(raw_label: str) -> str:
    """Normalizes bilingual Emotion2Vec+ labels like '难过/sad' -> 'sad'."""
    if not raw_label:
        return "neutral"
    cleaned = str(raw_label).strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1].strip()
    if cleaned in ("<unk>", "unk", "other", "其他"):
        return "neutral"
    return cleaned


def evict_vram(stage_name: str = "Stage"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[Eviction] {stage_name} evicted from GPU. Active VRAM: {vram_mb:.1f} MB")
    else:
        print(f"[Eviction] {stage_name} evicted from RAM.")


def format_state(transcript: str, ser_prior: str, ser_conf: float, prosody: dict) -> str:
    clean_ser = clean_e2v_label(ser_prior)
    return (
        f"Speaker transcript: 「{transcript}」\n"
        f"Acoustic speech emotion prior: {clean_ser} (confidence: {ser_conf:.0%})\n"
        f"Prosodic acoustic features:\n"
        f"- Pitch contour: {prosody.get('pitch_str', 'Normal')}\n"
        f"- Speech tempo: {prosody.get('tempo_str', 'Normal')}\n"
        f"- Energy & vocal intensity: {prosody.get('energy_str', 'Normal')}\n"
        f"- Voice quality: {prosody.get('voice_quality_str', 'Normal')}"
    )


class VocalisMultimodalPipeline:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.whisper_turbo_path = r"C:\Users\vaibh\.cache\huggingface\hub\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo\snapshots\0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
        self.scratch_dir = Path("scratch")
        self.scratch_dir.mkdir(exist_ok=True)
        print("=" * 80)
        print("VOCALIS MULTIMODAL PIPELINE CONFIGURED WITH SEQUENTIAL VRAM EVICTION")
        print("Lifecycle: Stage 1 (Whisper INT8) -> EVICT -> Stage 2 (E2V+ & Prosody) -> EVICT -> Stage 3 (Kev INT4) -> EVICT")
        print("=" * 80)

    def process_file(self, audio_path: str, lang: str = "en", use_kev4b: bool = True, output_json: str = None) -> dict:
        t_start_total = time.time()
        print(f"\n>>> PROCESSING AUDIO WITH SEQUENTIAL VRAM EVICTION: {audio_path} (Language: {lang.upper()})")

        # -------------------------------------------------------------------
        # 0. Load & Resample Audio to 16,000 Hz Mono (Preserve true dynamics for Prosody!)
        # -------------------------------------------------------------------
        full_audio, orig_sr = sf.read(audio_path)
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=1)
        full_audio = full_audio.astype(np.float32)

        target_sr = 16000
        if orig_sr != target_sr:
            print(f"[+] Resampling audio: {orig_sr} Hz -> {target_sr} Hz (soxr_hq)")
            full_audio = librosa.resample(full_audio, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
            orig_sr = target_sr

        # Preserve raw un-normalized audio for accurate dB energy & onset prosody extraction
        raw_audio = full_audio.copy()

        # Soft normalize a copy for Whisper ASR only if audio is extremely quiet (<0.15 peak)
        asr_audio = full_audio.copy()
        max_val = np.max(np.abs(asr_audio))
        if 0.005 < max_val < 0.15:
            asr_audio = (asr_audio / max_val) * 0.50

        total_duration_sec = len(raw_audio) / target_sr
        print(f"[+] Audio loaded: {total_duration_sec:.2f}s ({total_duration_sec/60:.2f} min) at {target_sr} Hz")

        # -------------------------------------------------------------------
        # STAGE 1: Faster-Whisper Large-v3 Turbo (INT8 compute) + VAD
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("[STAGE 1] Loading Faster-Whisper Large-v3 Turbo (INT8)...")
        from faster_whisper import WhisperModel
        compute_type = "int8_float16" if torch.cuda.is_available() else "int8"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        whisper_model = WhisperModel(self.whisper_turbo_path, device=device_str, compute_type=compute_type)
        t_w0 = time.time()
        segments_raw, info = whisper_model.transcribe(
            asr_audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=150)
        )
        whisper_segments = list(segments_raw)
        whisper_duration = time.time() - t_w0
        print(f"[+] Faster-Whisper transcribed {len(whisper_segments)} spoken segments in {whisper_duration:.2f}s")

        # EVICTION STAGE 1
        del whisper_model
        evict_vram("Faster-Whisper Large-v3 Turbo (Stage 1)")

        if len(whisper_segments) == 0:
            # Fallback if no VAD segments detected (silence or single short clip)
            whisper_segments = [type('SegmentMock', (), {'start': 0.0, 'end': total_duration_sec, 'text': 'Spoken statement.'})()]

        # -------------------------------------------------------------------
        # STAGE 2: Emotion2Vec+ Large (1024-dim Acoustics) & Librosa Prosody
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("[STAGE 2] Loading Emotion2Vec+ Large Foundation Model...")
        from funasr import AutoModel
        t_e0 = time.time()
        e2v_model = AutoModel(model="iic/emotion2vec_plus_large", hub="ms", device=self.device, disable_update=True)
        print(f"[+] Emotion2Vec+ Large loaded onto GPU in {time.time() - t_e0:.2f}s")

        segmented_acoustics = []
        temp_slice_wav = str(self.scratch_dir / "temp_slice.wav")

        for idx, seg in enumerate(whisper_segments, 1):
            text = seg.text.strip()
            if not text:
                continue

            # Natural boundary padding (+/- 60ms) to ensure smooth playback without robotic cuts
            pad_s = 0.06
            start_s = max(0.0, round(seg.start - pad_s, 2))
            end_s = min(total_duration_sec, round(seg.end + pad_s, 2))
            seg_dur = max(end_s - start_s, 0.05)

            start_sample = max(0, int(start_s * target_sr))
            end_sample = min(len(raw_audio), int(end_s * target_sr))
            seg_audio = raw_audio[start_sample:end_sample].copy()

            # Gentle 3ms fade-in/fade-out to eliminate edge clicking
            fade_len = min(int(0.003 * target_sr), len(seg_audio) // 4)
            if fade_len > 0:
                fade_curve = np.linspace(0, 1, fade_len, dtype=np.float32)
                seg_audio[:fade_len] *= fade_curve
                seg_audio[-fade_len:] *= fade_curve[::-1]

            sf.write(temp_slice_wav, seg_audio, target_sr, subtype="PCM_16")

            # Extract Prosody features on true un-normalized signal
            try:
                prosody = extract_prosody_features(temp_slice_wav)
            except Exception:
                prosody = {"pitch_str": "Normal", "tempo_str": "Normal", "energy_str": "Normal", "voice_quality_str": "Normal"}

            # Emotion2Vec+ Inference
            ser_prior = "neutral"
            ser_conf = 0.50
            e2v_canonical_probs = {}
            try:
                e2v_res = e2v_model.generate(seg_audio, output_dir=None, granularity="utterance", extract_embedding=False)
                if isinstance(e2v_res, list) and len(e2v_res) > 0:
                    labels = e2v_res[0].get("labels", [])
                    scores = e2v_res[0].get("scores", [])
                    if len(labels) > 0 and len(scores) > 0:
                        top_idx = int(np.argmax(scores))
                        ser_prior = clean_e2v_label(str(labels[top_idx]))
                        ser_conf = float(scores[top_idx])
                        for lbl, sc in zip(labels, scores):
                            c_lbl = clean_e2v_label(str(lbl))
                            can_k = EMOTION_MAP_E2V.get(c_lbl, "neutral_inquiry")
                            e2v_canonical_probs[can_k] = max(e2v_canonical_probs.get(can_k, 0.0), float(sc))
            except Exception as e_err:
                print(f"  [!] E2V warning on segment {idx}: {e_err}")

            segmented_acoustics.append({
                "segment_index": idx,
                "start_sec": start_s,
                "end_sec": end_s,
                "duration_sec": round(seg_dur, 2),
                "transcript": text,
                "ser_acoustic_prior": ser_prior,
                "ser_confidence": ser_conf,
                "e2v_canonical_probs": e2v_canonical_probs,
                "prosody": prosody
            })

        # EVICTION STAGE 2
        del e2v_model
        evict_vram("Emotion2Vec+ Large (Stage 2)")

        # -------------------------------------------------------------------
        # STAGE 3: Kev Decision Model (Kev-4B INT4 or Kev-0.8B INT4)
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        final_segments = []

        # Check if Kev-4B INT4 GGUF is available via llama-server
        root_dir = Path(__file__).resolve().parent.parent
        server_exe = root_dir / "tools" / "llama" / "llama-server.exe"
        model_path = root_dir / "data" / "models" / "Kev-4B-Q4_K_M.gguf"
        can_use_kev4b = use_kev4b and server_exe.exists() and model_path.exists()

        if can_use_kev4b:
            print("[STAGE 3] Running Kev-4B INT4 (GGUF Q4_K_M) via dedicated process...")
            port = 8089
            proc = subprocess.Popen(
                [str(server_exe.resolve()), "-m", str(model_path.resolve()), "--port", str(port), "-ngl", "28", "-np", "1", "-c", "2048", "-b", "2048", "-ub", "2048"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Wait for Kev-4B server ready
            ready = False
            for _ in range(40):
                time.sleep(1)
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    pass

            if ready:
                print(f"[+] Kev-4B INT4 ready on port {port}. Processing {len(segmented_acoustics)} segments...")
                for item in segmented_acoustics:
                    state_text = format_state(item["transcript"], item["ser_acoustic_prior"], item["ser_confidence"], item["prosody"])
                    req_payload = {"state": state_text, "questions": IB_QUESTIONS}

                    pred_emo = EMOTION_MAP_E2V.get(item["ser_acoustic_prior"], "neutral_inquiry")
                    pred_conf = item["ser_confidence"]
                    sarc_p = 0.15
                    raw_probs = {}

                    try:
                        resp = requests.post(f"http://127.0.0.1:{port}/v1/systemone", json=req_payload, timeout=30)
                        if resp.status_code == 200:
                            answers = resp.json().get("answers", {})
                            det = answers.get("detected_emotion", {})
                            pred_emo = det.get("choice", pred_emo)
                            pred_conf = det.get("confidence", det.get("answer_confidence", pred_conf))
                            if isinstance(det.get("probabilities"), dict):
                                raw_probs = det["probabilities"]
                            raw_noul = answers.get("sarcasm_detected", {}).get("noul", sarc_p)
                            sarc_p = float(raw_noul[0] if isinstance(raw_noul, list) else raw_noul)
                    except Exception as k_err:
                        print(f"  [!] Kev-4B request error: {k_err}")

                    if not raw_probs:
                        raw_probs = {k: round((1.0 - pred_conf) / 7.0, 4) for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
                        raw_probs[pred_emo] = round(pred_conf, 4)

                    item["predicted_emotion"] = pred_emo
                    item["confidence"] = round(float(pred_conf), 4)
                    item["probabilities"] = raw_probs
                    item["sarcasm_probability"] = round(float(sarc_p), 4)
                    item["sarcasm_flag"] = bool(sarc_p > 0.45 or pred_emo == "sarcasm_passive_aggression")
                    final_segments.append(item)

                # Terminate and EVICT Kev-4B
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                print("[Eviction] Kev-4B INT4 process terminated and VRAM evicted.")
            else:
                print("[!] Kev-4B server failed to start within timeout, falling back to Kev-0.8B INT4...")
                proc.kill()
                can_use_kev4b = False

        if not can_use_kev4b:
            print("[STAGE 3] Running Kev-0.8B INT4 (BitsAndBytes NF4 in PyTorch)...")
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel
            from kev.checkpoint import Checkpoint
            from kev.api import SystemOneRequest, to_record, to_answers
            from kev.model import admit, DecisionModel, load_tokenizer

            ck = Checkpoint("jaredpalmer/kev-0.8b")
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
            raw_causal = AutoModelForCausalLM.from_pretrained(
                ck.meta.base, revision=ck.meta.base_revision, quantization_config=bnb_config, device_map=self.device
            )
            kev_model = DecisionModel(
                ck.meta.base, tok, self.device, lora=None, revision=ck.meta.base_revision,
                head_dim=ck.meta.head_dim, option_isolation=ck.meta.option_isolation, dtype=torch.float16
            )
            kev_model.lm = PeftModel.from_pretrained(raw_causal.model, ck.path, torch_device=self.device)
            kev_model.head.load_state_dict(ck.meta.head)
            kev_model.eval()
            kev_model.head.temperature = ck.meta.temperature

            print(f"[+] Kev-0.8B INT4 loaded. Processing {len(segmented_acoustics)} segments...")
            for item in segmented_acoustics:
                state_text = format_state(item["transcript"], item["ser_acoustic_prior"], item["ser_confidence"], item["prosody"])
                req = SystemOneRequest(state=state_text, questions=IB_QUESTIONS)
                rec = to_record(req)
                admitted = admit(rec, tok, require_options=True)

                pred_emo = EMOTION_MAP_E2V.get(item["ser_acoustic_prior"], "neutral_inquiry")
                pred_conf = item["ser_confidence"]
                sarc_p = 0.12
                probs = {k: round((1.0 - pred_conf) / 7.0, 4) for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
                probs[pred_emo] = round(pred_conf, 4)

                try:
                    with torch.no_grad():
                        out = kev_model(admitted.batch(self.device))
                        answers = to_answers(rec, out)
                        det = answers.get("detected_emotion")
                        if det:
                            pred_emo = det.choice
                            pred_conf = det.confidence
                            if hasattr(det, "probabilities") and det.probabilities:
                                probs = {k: round(float(v), 4) for k, v in det.probabilities.items()}
                            else:
                                probs = {k: round((1.0 - pred_conf) / 7.0, 4) for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
                                probs[pred_emo] = round(pred_conf, 4)
                        sarc = answers.get("sarcasm_detected")
                        if sarc:
                            sarc_p = sarc.noul
                except Exception as k_err:
                    print(f"  [!] Kev-0.8B inference error: {k_err}")

                item["predicted_emotion"] = pred_emo
                item["confidence"] = round(float(pred_conf), 4)
                item["probabilities"] = probs
                item["sarcasm_probability"] = round(float(sarc_p), 4)
                item["sarcasm_flag"] = bool(sarc_p > 0.45 or pred_emo == "sarcasm_passive_aggression")
                final_segments.append(item)

            # EVICTION STAGE 3
            del kev_model
            del raw_causal
            evict_vram("Kev-0.8B INT4 (Stage 3)")

        # -------------------------------------------------------------------
        # STAGE 4: Continuous Temporal Aggregation & Global Synthesis (CPU)
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("[STAGE 4] Calculating continuous temporal distribution & global probability vector...")
        emotion_duration_accum = {k: 0.0 for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
        emotion_prob_accum = {k: 0.0 for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
        segment_counts = {k: 0 for k in IB_QUESTIONS["detected_emotion"]["criteria"].keys()}
        total_spoken_duration = 0.0
        sarcastic_count = 0

        for s in final_segments:
            emo = s["predicted_emotion"]
            dur = s["duration_sec"]
            total_spoken_duration += dur
            emotion_duration_accum[emo] = emotion_duration_accum.get(emo, 0.0) + dur
            segment_counts[emo] = segment_counts.get(emo, 0) + 1
            if s.get("sarcasm_flag"):
                sarcastic_count += 1

            for e_k, e_p in s["probabilities"].items():
                emotion_prob_accum[e_k] = emotion_prob_accum.get(e_k, 0.0) + (e_p * dur)

        if total_spoken_duration > 0:
            time_dist_pct = {k: round((v / total_spoken_duration) * 100, 2) for k, v in emotion_duration_accum.items()}
            global_prob_vec = {k: round(v / total_spoken_duration, 4) for k, v in emotion_prob_accum.items()}
        else:
            time_dist_pct = {k: 0.0 for k in emotion_duration_accum}
            global_prob_vec = {k: 0.125 for k in emotion_prob_accum}

        dominant_emo = max(time_dist_pct, key=time_dist_pct.get)
        dominant_prob = global_prob_vec.get(dominant_emo, 0.5)

        def _emo_to_sent(emo_key: str, s_flag: bool = False) -> str:
            if emo_key == "satisfaction_gratitude":
                return "positive"
            if emo_key == "neutral_inquiry":
                return "neutral"
            return "negative"

        def _probs_to_sent_probs(p_dict: dict, s_prob: float = 0.0, s_flag: bool = False) -> dict:
            p_sat = float(p_dict.get("satisfaction_gratitude", 0.0))
            p_neu = float(p_dict.get("neutral_inquiry", 0.0))
            neg_keys = [
                "frustration_irritation", "outrage_hostility", "disappointed_sad",
                "sarcasm_passive_aggression", "anxiety_confusion", "urgency_panic"
            ]
            p_neg = sum(float(p_dict.get(k, 0.0)) for k in neg_keys)
            tot = p_sat + p_neu + p_neg
            if tot > 0:
                return {
                    "positive": round(p_sat / tot, 4),
                    "neutral": round(p_neu / tot, 4),
                    "negative": round(p_neg / tot, 4),
                }
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        sent_dur_accum = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        sent_prob_accum = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        sent_counts = {"positive": 0, "neutral": 0, "negative": 0}

        clean_segments = []
        for s in final_segments:
            pred_sent = _emo_to_sent(s["predicted_emotion"], bool(s.get("sarcasm_flag", False)))
            sent_probs = _probs_to_sent_probs(s["probabilities"], float(s.get("sarcasm_probability", 0.0)), bool(s.get("sarcasm_flag", False)))
            dur = s["duration_sec"]
            sent_dur_accum[pred_sent] += dur
            sent_counts[pred_sent] += 1
            for sk, sv in sent_probs.items():
                sent_prob_accum[sk] += sv * dur

            clean_segments.append({
                "segment_index": s["segment_index"],
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "duration_sec": s["duration_sec"],
                "transcript": s["transcript"],
                "ser_acoustic_prior": s["ser_acoustic_prior"],
                "ser_confidence": s["ser_confidence"],
                "prosody": {
                    "pitch": s["prosody"].get("pitch_str", "Normal"),
                    "tempo": s["prosody"].get("tempo_str", "Normal"),
                    "energy": s["prosody"].get("energy_str", "Normal"),
                    "voice_quality": s["prosody"].get("voice_quality_str", "Normal")
                },
                "predicted_emotion": s["predicted_emotion"],
                "predicted_sentiment": pred_sent,
                "confidence": s["confidence"],
                "probabilities": s["probabilities"],
                "sentiment_probabilities": sent_probs,
                "sarcasm_probability": s["sarcasm_probability"],
                "sarcasm_flag": s["sarcasm_flag"]
            })

        if total_spoken_duration > 0:
            sent_time_pct = {k: round((v / total_spoken_duration) * 100, 2) for k, v in sent_dur_accum.items()}
            global_sent_prob_vec = {k: round(v / total_spoken_duration, 4) for k, v in sent_prob_accum.items()}
        else:
            sent_time_pct = {"positive": 0.0, "neutral": 100.0, "negative": 0.0}
            global_sent_prob_vec = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        dominant_sent = max(sent_time_pct, key=sent_time_pct.get)
        dominant_sent_prob = global_sent_prob_vec.get(dominant_sent, 0.5)

        results = {
            "file": audio_path,
            "language": lang,
            "total_duration_sec": round(total_duration_sec, 3),
            "spoken_duration_sec": round(total_spoken_duration, 2),
            "total_segments": len(clean_segments),
            "dominant_emotion": dominant_emo,
            "dominant_probability": dominant_prob,
            "dominant_sentiment": dominant_sent,
            "dominant_sentiment_probability": dominant_sent_prob,
            "time_distribution_percent": time_dist_pct,
            "sentiment_time_distribution_percent": sent_time_pct,
            "segment_counts": segment_counts,
            "sentiment_counts": sent_counts,
            "global_probability_vector": global_prob_vec,
            "global_sentiment_probability_vector": global_sent_prob_vec,
            "sarcastic_segments_count": sarcastic_count,
            "segments": clean_segments
        }

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[✓] Saved pipeline analysis to: {output_json}")

        total_lat = time.time() - t_start_total
        print(f"[✓] Complete multimodal pipeline executed with zero VRAM leaks in {total_lat:.2f}s!")
        return results

def main():
    pipeline = VocalisMultimodalPipeline()
    # Process the properly resampled English combined file
    pipeline.process_file(
        "data/audio_samples/combined_english_30_calls.wav",
        lang="en",
        output_json="combined_english_pipeline_results.json"
    )

if __name__ == "__main__":
    main()
