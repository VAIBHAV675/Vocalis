# 🧠 Kev `SystemOne` Prompt Structure & Tri-Modal Priming Architecture

This document explains the exact prompt structure sent to **`Kev-0.8B` / `Kev-4B`** in [`backend/vocalis_segmented_pipeline.py`](../backend/vocalis_segmented_pipeline.py) and [`scripts/benchmarks/run_2000_kev_pipeline_eval.py`](../scripts/benchmarks/run_2000_kev_pipeline_eval.py), why **Tri-Modal Prompt Priming** is used instead of post-hoc score fusion, and why **both** the `Emotion2Vec+` neural SER prior and `Librosa` physical prosody are necessary.

---

## 1. Why Prompt Priming Instead of Post-LLM Score Fusion?

1. **Benchmark Consistency:** All 60-clip and 2,000-clip evaluations in [`benchmarks/results/`](../benchmarks/results/) use **Tri-Modal Prompt Priming**, where `Kev`'s `PointerHead` (`choice`) and `noul` (`sarcasm_detected`) heads directly read the transcript, the `Emotion2Vec+` acoustic prior, and the `Librosa` prosodic descriptors in a single forward pass—with **zero post-LLM `if/else` rules or post-hoc score averaging**.
2. **Non-Linear Cross-Modal Reasoning:** As shown in [`scripts/experiments/run_fusion_comparison_60.py`](../scripts/experiments/run_fusion_comparison_60.py) (`43.3%` accuracy for static linear fusion vs. `80.0%` for Kev priming), linear probability averaging ($w_1 \mathbf{p}_{\text{text}} + w_2 \mathbf{p}_{\text{audio}}$) cannot perform non-linear XOR reasoning (such as detecting that *Positive Words* + *Sad/Hostile Voice* = *Sarcasm / Contradiction* via the `sarcasm_detected` `noul` head).

---

## 2. Why Do We Need BOTH the `Emotion2Vec+` SER Prior AND `Librosa` Physical Prosody?

A natural question is: *"If we already pass Librosa physical prosody (Pitch in Hz, Energy in dB, Tempo in onsets/sec, Voice Quality), why do we also need the `Emotion2Vec+` SER prior?"* (or vice versa).

Empirical ablations across the dataset proved that **neither signal works alone**:

| Acoustic Input Provided to LLM | Accuracy (`60` Audited) | Accuracy (`2,000` Scale) | Why It Fails Without the Other |
| :--- | :---: | :---: | :--- |
| **1. `Librosa` Physical Prosody ONLY (No SER Prior)**<br/>([`run_pure_prosody_priming_eval.py`](../scripts/experiments/run_pure_prosody_priming_eval.py)) | **13.3%** (`8/60`) | — | **Physiological Baseline Ambiguity:** Raw numbers like `"Pitch: 115 Hz (flat contour), Energy: 66 dB, deep chest resonance"` describe a normal, calm adult male voice—yet without a neural SER prior, text LLMs misinterpret low pitch + moderate volume as `disappointed_sad` on >80% of speakers. `Emotion2Vec+` uses a 1024-dim representation trained on 40,000+ hours of speech to detect micro-tremors, formant shifts, and vocal tract tension that simple Hz/dB summaries cannot capture. |
| **2. `Emotion2Vec+` SER Prior ONLY (No LLM + Prosody)**<br/>([`run_2000_kev_pipeline_eval.py`](../scripts/benchmarks/run_2000_kev_pipeline_eval.py)) | **61.7%** (`37/60`) | **53.15%** (`1063/2000`) | **Prosodic Broadcast Hallucination:** On Japanese (`ReazonSpeech`), animated pitch modulation causes `Emotion2Vec+` to hallucinate `happy` on **52.2%** (`522/1,000`) of clips (`26.1%` JA accuracy). Without `Librosa` prosody (`dB` intensity, onset rate, pitch slope) and the transcript, the system cannot tell animated news reading apart from genuine joy. |
| **3. Combined Tri-Modal Priming (`Transcript` + `E2V+ Prior` + `Librosa Prosody`)** | **80.0%** (`48/60`) | **65.40%** (`1308/2000`) | **Complementary Grounding:** `Emotion2Vec+` provides the high-dimensional emotional prior (`sad`, `angry`, `happy`), `Librosa` provides objective physical measurements (`Hz`, `dB`, `onsets/sec`) to verify or temper that prior, and `Faster-Whisper` provides the semantic context. |

---

## 3. Final Kev `SystemOne` Prompt Structure

### A. The `state` String (`format_state` in [`backend/vocalis_segmented_pipeline.py`](../backend/vocalis_segmented_pipeline.py))

```python
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
```

#### Example Rendered `state` (from `rec_1791807581.wav`):
```text
Speaker transcript: 「I'm very happy in my life. Life is doing very good.」
Acoustic speech emotion prior: sad (confidence: 82%)
Prosodic acoustic features:
- Pitch contour: 193.0 Hz (std: 49.0 Hz, downward falling terminal contour)
- Speech tempo: 3.4 onsets/sec (slow / measured / subdued pace)
- Energy & vocal intensity: 73.5 dB (moderate baseline business volume)
- Voice quality: standard modal phonation, clean harmonic structure
```

---

### B. The `questions` Schema (`IB_QUESTIONS` in [`backend/vocalis_segmented_pipeline.py`](../backend/vocalis_segmented_pipeline.py))

`Kev` evaluates two simultaneous heads in a single forward pass:
1. **`detected_emotion` (`type: "choice"`):** Uses the learned `PointerHead` (`dp=256`) with option isolation to score all 8 canonical classes simultaneously.
2. **`sarcasm_detected` (`type: "noul"`):** Uses the binary `noul` head to output a calibrated probability $\in [0, 1]$ measuring whether the speaker's literal words contradict their vocal tone.

```json
{
  "detected_emotion": {
    "type": "choice",
    "instructions": "Analyze the caller's spoken statement in the full context of their vocal acoustics and delivery. Resolve any tension between the words and vocal tone (e.g. sarcasm, polite complaints, subdued relief). Classify the customer's true emotional state into one of the 8 canonical categories.",
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
```

---

### C. Example Raw `/v1/systemone` Response from `Kev-4B` (Fixes #1 & #2 Only)

When fed the rendered `state` and `questions` above on `rec_1791807581.wav`:

```json
{
  "answers": {
    "detected_emotion": {
      "choice": "satisfaction_gratitude",
      "confidence": 0.3683,
      "probabilities": {
        "satisfaction_gratitude": 0.4472,
        "neutral_inquiry": 0.2391,
        "disappointed_sad": 0.1900,
        "sarcasm_passive_aggression": 0.0778,
        "anxiety_confusion": 0.0230,
        "frustration_irritation": 0.0133,
        "urgency_panic": 0.0054,
        "outrage_hostility": 0.0041
      }
    },
    "sarcasm_detected": {
      "noul": 0.5715
    }
  }
}
```
