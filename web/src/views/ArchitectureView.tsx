import React, { useState } from 'react';
import {
  Zap,
  CheckCircle2,
  ShieldAlert,
  Sparkles,
  Layers,
  BookOpen,
  Gauge,
  Award,
  FlaskConical,
} from 'lucide-react';

export const ArchitectureView: React.FC = () => {
  const [activeStage, setActiveStage] = useState<number>(1);

  const stages = [
    {
      num: 1,
      title: 'Faster-Whisper Large-v3 Turbo (INT8)',
      category: 'ASR & Temporal Segmentation',
      model: 'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
      precision: 'int8_float16 (CUDA) / int8 (CPU)',
      desc: 'High-throughput multilingual speech-to-text with integrated Silero Voice Activity Detection (VAD). Slices long continuous audio into discrete spoken segments with ±60ms padding and 3ms anti-click fade envelopes.',
      metrics: 'EN: 2,174.8 ms (0.3565x RTF, 2.8x faster) | JA: 2,470.6 ms (0.3437x RTF, 2.9x faster) | 1.1 GB VRAM',
    },
    {
      num: 2,
      title: 'Emotion2Vec+ Large (1024-dim)',
      category: 'Acoustic Speech Foundation',
      model: 'iic/emotion2vec_plus_large (ModelScope / HF)',
      precision: 'FP16 / FP32',
      desc: 'Foundation acoustic encoder trained on 40,000+ hours of emotional speech. Generates rich 1024-dimensional continuous latent representations and a 7-class acoustic emotion prior before evicting from VRAM.',
      metrics: '203 ms/clip (~30x faster than real-time) | 1.24 GB Active VRAM | 83.3% EN standalone prior',
    },
    {
      num: 3,
      title: 'Biomechanical Prosody Engine',
      category: 'Acoustic Feature Extraction',
      model: 'Librosa Sound Processing Suite',
      precision: 'Analytical / Vectorized DSP',
      desc: 'Quantifies physiological speech dynamics: Fundamental Frequency contour via YIN (F0 in Hz & terminal slope), Speech Tempo (onsets/sec), Vocal Intensity RMS (dB), and Phonation Quality (spectral centroid & flatness).',
      metrics: '~58 ms/clip on CPU (~105x faster than real-time) | 0 MB GPU VRAM | Deterministic physical features',
    },
    {
      num: 4,
      title: 'Kev-0.8B / Kev-4B Pre-Quantized Decision',
      category: 'Multimodal Cognitive Arbitration',
      model: 'ggml-org/Kev-0.8B-GGUF (Q8_0) & Kev-4B-GGUF (Q4_K_M)',
      precision: 'Pre-Quantized GGUF / BitsAndBytes NF4',
      desc: 'Built on a Qwen3.5 hybrid Gated DeltaNet backbone with a block-causal branch mask and learned PointerHead (dp=256). Scores all 8 canonical emotion options in a single forward pass with zero token generation.',
      metrics: 'Kev-0.8B: 545–640 ms/clip (1.51 clips/s C++ CUDA, 803 MB VRAM) | Kev-4B: 2,853–3,154 ms/clip (2.41 GB VRAM)',
    },
    {
      num: 5,
      title: 'Continuous Temporal Integration',
      category: 'Duration-Weighted Synthesis',
      model: 'Vocalis Time-Weighted Integration',
      precision: 'Mathematical Formulation',
      desc: 'Aggregates segment predictions into a unified continuous call profile. Weights emotional states by spoken segment duration rather than count, preventing brief 0.5s filler interjections from skewing call analytics.',
      metrics: 'Continuous global probability vector + TimeShare(%) across 8 canonical emotions',
    },
    {
      num: 6,
      title: 'Acoustic Incongruity & Sarcasm Filter',
      category: 'Friction & Risk Heuristic',
      model: 'Divergence Metric & Kev Binary Sarcasm Head',
      precision: 'Dual-Head Cross-Validation',
      desc: 'Detects divergence between polite lexical sentiment and flat/strained vocal acoustic delivery. Flags corporate passive-aggression where formal phrasing ("Thank you for that clarification") conceals sharp frustration.',
      metrics: 'Dedicated noul binary sarcasm head + acoustic-lexical contradiction flagging',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl">
        <div className="flex items-center space-x-2 mb-1">
          <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
            <BookOpen className="w-4 h-4" />
          </span>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            Vocalis Multimodal Architecture, Speed & Accuracy Benchmarks
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-400">
          Sequential VRAM-evicted multimodal architecture running Faster-Whisper Large-v3 Turbo INT8, Emotion2Vec+ Large, Librosa DSP, and Pre-Quantized Kev-0.8B / Kev-4B within a strict 4 GB GPU VRAM envelope.
        </p>
      </div>

      {/* 6-Stage Interactive Architecture Flow */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-slate-200 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>Interactive 6-Stage Pipeline & Sequential VRAM Eviction Lifecycle</span>
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {stages.map((st) => (
            <button
              key={st.num}
              onClick={() => setActiveStage(st.num)}
              className={`p-3.5 rounded-2xl border text-left transition-all ${
                activeStage === st.num
                  ? 'bg-indigo-950/80 border-indigo-500 shadow-lg shadow-indigo-500/20 ring-1 ring-indigo-500'
                  : 'bg-slate-900/80 border-slate-800 hover:bg-slate-900 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="w-6 h-6 rounded-lg bg-indigo-600/30 text-indigo-300 font-mono text-xs flex items-center justify-center font-bold">
                  0{st.num}
                </span>
                {activeStage === st.num && <Sparkles className="w-3.5 h-3.5 text-indigo-400" />}
              </div>
              <span className="text-xs font-semibold text-white block truncate mb-1">
                {st.title.split('(')[0]}
              </span>
              <span className="text-[10px] text-slate-400 block truncate font-mono">
                {st.category}
              </span>
            </button>
          ))}
        </div>

        {/* Detailed Stage Card */}
        {(() => {
          const st = stages.find((s) => s.num === activeStage)!;
          return (
            <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4 animate-fadeIn">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div>
                  <span className="text-xs font-mono text-indigo-400 uppercase tracking-wider block mb-1">
                    Stage 0{st.num} • {st.category}
                  </span>
                  <h3 className="text-lg font-bold text-white">{st.title}</h3>
                </div>
                <div className="flex items-center space-x-2 text-xs">
                  <span className="px-2.5 py-1 rounded-full bg-slate-950 border border-slate-800 font-mono text-slate-300">
                    {st.precision}
                  </span>
                </div>
              </div>

              <p className="text-sm text-slate-300 leading-relaxed">{st.desc}</p>

              <div className="p-3 rounded-2xl bg-slate-950/80 border border-slate-800 flex items-center space-x-2 text-xs text-indigo-300 font-mono">
                <Zap className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                <span>Performance & Telemetry: {st.metrics}</span>
              </div>
            </div>
          );
        })()}
      </div>

      {/* 1. Extended 2,000-Clip Large-Scale Benchmark Matrix */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Award className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-semibold text-white">
              1. Extended 2,000-Clip Multilingual Benchmark (1,000 EN Earnings22 + 1,000 JA ReazonSpeech)
            </h2>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 px-2.5 py-1 rounded-full">
            +21.20% JA Gain over Standalone Acoustic SER
          </span>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed">
          Evaluated across 2,000 real corporate earnings calls and Japanese broadcasts. Standalone <code className="text-indigo-300">Emotion2Vec+ Large</code> collapses to 26.10% on Japanese due to misclassifying animated broadcast cadence as <code className="text-amber-300">happy</code> on 52.2% (522/1,000) of clips; multimodal arbitration with <code className="text-indigo-300">Kev-0.8B</code> nearly doubles Japanese accuracy to 47.30%.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Pipeline Architecture</th>
                <th className="py-3 px-4">Pre-Quantized Checkpoint</th>
                <th className="py-3 px-4 text-center">EN 8-Emo / Sentiment</th>
                <th className="py-3 px-4 text-center">JA 8-Emo / Sentiment</th>
                <th className="py-3 px-4 text-center">Overall 8-Emo</th>
                <th className="py-3 px-4 text-center text-emerald-400">Overall 3-Sentiment</th>
                <th className="py-3 px-4 text-right">C++ CUDA Speed</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-xs">
              <tr className="bg-emerald-950/20 font-semibold text-white">
                <td className="py-3.5 px-4 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span>Best Per-Language Combination (4B EN + 0.8B JA)</span>
                </td>
                <td className="py-3.5 px-4 text-indigo-300">Pre-Quantized GGUF</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">83.5% / 84.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">47.3% / 50.9% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">65.40% (1308/2000)</td>
                <td className="py-3.5 px-4 text-center text-emerald-300 font-bold">67.60% (1352/2000)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">640–3,154 ms/clip</td>
              </tr>
              <tr className="bg-indigo-950/20 font-semibold text-white">
                <td className="py-3.5 px-4">Kev-0.8B Multimodal Pipeline</td>
                <td className="py-3.5 px-4 text-indigo-300">ggml-org/Kev-0.8B-GGUF (Q8_0)</td>
                <td className="py-3.5 px-4 text-center text-slate-200">83.0% / 83.4% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">47.3% / 50.9% Sent</td>
                <td className="py-3.5 px-4 text-center text-indigo-300 font-bold">65.15% (1303/2000)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">67.15% (1343/2000)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">640 ms (1.51 clips/s)</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Kev-4B Multimodal Pipeline</td>
                <td className="py-3.5 px-4 text-slate-400">ggml-org/Kev-4B-GGUF (Q4_K_M INT4)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">83.5% / 84.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-300">44.2% / 47.7% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-200">63.85% (1277/2000)</td>
                <td className="py-3.5 px-4 text-center text-indigo-300">66.00% (1320/2000)</td>
                <td className="py-3.5 px-4 text-right text-slate-300">3,154 ms (0.31 clips/s)</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 text-slate-400">Emotion2Vec+ Large (Acoustic SER Standalone)</td>
                <td className="py-3.5 px-4 text-slate-500">FP16 (iic/emotion2vec_plus_large)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">80.2% / 80.8% Sent</td>
                <td className="py-3.5 px-4 text-center text-rose-400">26.1% / 29.5% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">53.15% (1063/2000)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">55.15% (1103/2000)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">203 ms/clip</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Initial 60-Clip Audited Benchmark Matrix */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center space-x-2">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          <h2 className="text-base font-semibold text-white">
            2. Human-Audited 60-Clip Reference Benchmark (30 EN Earnings22 + 30 JA ReazonSpeech)
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Precision / Quant</th>
                <th className="py-3 px-4 text-center">EN 8-Emo / Sentiment</th>
                <th className="py-3 px-4 text-center">JA 8-Emo / Sentiment</th>
                <th className="py-3 px-4 text-center">Overall 8-Emo (60)</th>
                <th className="py-3 px-4 text-center text-emerald-400">Overall 3-Sentiment (60)</th>
                <th className="py-3 px-4 text-right">C++ CUDA Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-xs">
              <tr className="bg-emerald-950/20 font-semibold text-white">
                <td className="py-3.5 px-4 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span>Vocalis Language-Routed Production (0.8B EN + 4B JA)</span>
                </td>
                <td className="py-3.5 px-4 text-indigo-300">Pre-Quantized</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">93.3% / 96.7% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">66.7% / 76.7% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">80.0% (48/60)</td>
                <td className="py-3.5 px-4 text-center text-emerald-300 font-bold">86.7% (52/60)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">~1,700 ms avg</td>
              </tr>
              <tr className="bg-indigo-950/20 font-semibold text-white">
                <td className="py-3.5 px-4">Kev-0.8B Pre-Quantized (EN Champion)</td>
                <td className="py-3.5 px-4 text-indigo-300">NF4 / GGUF Q8_0</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">93.3% / 96.7% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-300">43.3% / 53.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-indigo-300">68.3% (41/60)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400">75.0% (45/60)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">545 ms (C++ CUDA)</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Kev-4B Pre-Quantized (JA Champion)</td>
                <td className="py-3.5 px-4 text-slate-300">GGUF Q4_K_M</td>
                <td className="py-3.5 px-4 text-center text-slate-300">86.7% / 90.0% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">66.7% / 76.7% Sent</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">76.7% (46/60)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">83.3% (50/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-300">2,853 ms (C++ CUDA)</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Kev-0.8B Multimodal Pipeline (FP16)</td>
                <td className="py-3.5 px-4 text-slate-400">FP16 Merged</td>
                <td className="py-3.5 px-4 text-center text-slate-300">90.0% / 93.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">46.7% / 53.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-300">68.3% (41/60)</td>
                <td className="py-3.5 px-4 text-center text-slate-200">73.3% (44/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">2,362 ms (PyTorch)</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Emotion2Vec+ Large Standalone</td>
                <td className="py-3.5 px-4 text-slate-400">FP16</td>
                <td className="py-3.5 px-4 text-center text-slate-300">83.3% / 83.3% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">40.0% / 50.0% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">61.7% (37/60)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">66.7% (40/60)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">203 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Laya mmBERT Multimodal</td>
                <td className="py-3.5 px-4 text-slate-400">FP16</td>
                <td className="py-3.5 px-4 text-center text-slate-300">70.0% / 70.0% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">36.7% / 50.0% Sent</td>
                <td className="py-3.5 px-4 text-center text-slate-400">53.3% (32/60)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">60.0% (36/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">1,204 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. ASR & Hardware Speed Benchmarks */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Gauge className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-semibold text-white">
              3. ASR & Hardware Speed / Latency Benchmarks (NVIDIA GTX 1650 4 GB VRAM)
            </h2>
          </div>
          <span className="text-xs font-mono text-indigo-300 bg-indigo-950/50 border border-indigo-500/30 px-2.5 py-1 rounded-full">
            2.8x–2.9x Faster Than Real-Time (INT8-FP16)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">ASR Model Checkpoint</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4 text-center">Lang</th>
                <th className="py-3 px-4 text-right">Latency (6.1s–7.2s)</th>
                <th className="py-3 px-4 text-right">Real-Time Factor (RTF)</th>
                <th className="py-3 px-4 text-right">Speedup</th>
                <th className="py-3 px-4">4 GB GPU Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-xs">
              <tr className="bg-emerald-950/20 font-semibold text-white">
                <td className="py-3 px-4">whisper-large-v3-turbo (Selected)</td>
                <td className="py-3 px-4 text-indigo-300">int8_float16</td>
                <td className="py-3 px-4 text-center">EN</td>
                <td className="py-3 px-4 text-right text-emerald-400">2,174.8 ms</td>
                <td className="py-3 px-4 text-right text-emerald-400">0.3565x</td>
                <td className="py-3 px-4 text-right text-emerald-400">2.8x Faster</td>
                <td className="py-3 px-4 text-emerald-400">✅ 1.1 GB VRAM</td>
              </tr>
              <tr className="bg-emerald-950/20 font-semibold text-white">
                <td className="py-3 px-4">whisper-large-v3-turbo (Selected)</td>
                <td className="py-3 px-4 text-indigo-300">int8_float16</td>
                <td className="py-3 px-4 text-center">JA</td>
                <td className="py-3 px-4 text-right text-emerald-400">2,470.6 ms</td>
                <td className="py-3 px-4 text-right text-emerald-400">0.3437x</td>
                <td className="py-3 px-4 text-right text-emerald-400">2.9x Faster</td>
                <td className="py-3 px-4 text-emerald-400">✅ 1.1 GB VRAM</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-slate-400">whisper-large-v3-turbo</td>
                <td className="py-3 px-4 text-slate-500">float16</td>
                <td className="py-3 px-4 text-center">Multi</td>
                <td className="py-3 px-4 text-right text-rose-400">—</td>
                <td className="py-3 px-4 text-right text-rose-400">—</td>
                <td className="py-3 px-4 text-right text-rose-400">—</td>
                <td className="py-3 px-4 text-rose-400">❌ CUDA OOM (&gt;4 GB)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-slate-300">kotoba-whisper-v2.0</td>
                <td className="py-3 px-4 text-slate-400">float16</td>
                <td className="py-3 px-4 text-center">JA</td>
                <td className="py-3 px-4 text-right text-amber-400">8,699.1 ms</td>
                <td className="py-3 px-4 text-right text-amber-400">1.4260x</td>
                <td className="py-3 px-4 text-right text-amber-400">0.7x (Slower)</td>
                <td className="py-3 px-4 text-amber-400">⚠️ Repetition hallucinations</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-slate-300">kotoba-whisper-v2.0</td>
                <td className="py-3 px-4 text-slate-400">int8_float16</td>
                <td className="py-3 px-4 text-center">JA</td>
                <td className="py-3 px-4 text-right">2,226.6 ms</td>
                <td className="py-3 px-4 text-right">0.3650x</td>
                <td className="py-3 px-4 text-right">2.7x Faster</td>
                <td className="py-3 px-4 text-amber-400">⚠️ Requires 2nd model load</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-slate-300">whisper-small.en</td>
                <td className="py-3 px-4 text-slate-400">int8_float16</td>
                <td className="py-3 px-4 text-center">EN</td>
                <td className="py-3 px-4 text-right">916.0 ms</td>
                <td className="py-3 px-4 text-right">0.1502x</td>
                <td className="py-3 px-4 text-right">6.7x Faster</td>
                <td className="py-3 px-4 text-rose-400">❌ English-only</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Ablation Studies & Rejected Baselines Post-Mortem */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex items-center space-x-2">
          <FlaskConical className="w-5 h-5 text-pink-400" />
          <h2 className="text-base font-semibold text-white">
            4. Ablation Studies & Why Conventional SER / Fusion Approaches Failed
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Experiment / Baseline</th>
                <th className="py-3 px-4 text-center">English Acc</th>
                <th className="py-3 px-4 text-center">Japanese Acc</th>
                <th className="py-3 px-4 text-center">Overall Acc</th>
                <th className="py-3 px-4">Root Cause & Failure Analysis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono text-xs">
              <tr>
                <td className="py-3 px-4 text-rose-300">1. Pure Prosody Priming (No SER Prior)</td>
                <td className="py-3 px-4 text-center">13.3% (4/30)</td>
                <td className="py-3 px-4 text-center">13.3% (4/30)</td>
                <td className="py-3 px-4 text-center text-rose-400 font-bold">13.3% (8/60)</td>
                <td className="py-3 px-4 font-sans text-slate-400">
                  Raw physical metrics ("145 Hz, deep chest resonance") without an SER prior caused encoder routers to misclassify male chest resonance as <code className="text-slate-200">disappointed_sad</code>.
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-rose-300">2. ehcalabres/wav2vec2-lg-xlsr-en SER</td>
                <td className="py-3 px-4 text-center text-rose-400 font-bold">16.7% (5/30)</td>
                <td className="py-3 px-4 text-center">—</td>
                <td className="py-3 px-4 text-center text-rose-400">16.7% (5/30)</td>
                <td className="py-3 px-4 font-sans text-slate-400">
                  Fine-tuned on theatrical RAVDESS actors; softmax collapsed to uniform noise (~13.4% confidence) on real corporate earnings calls.
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-rose-300">3. Verbose Coherent Narrative Prompt</td>
                <td className="py-3 px-4 text-center">20.0% (6/30)</td>
                <td className="py-3 px-4 text-center">13.3% (4/30)</td>
                <td className="py-3 px-4 text-center text-rose-400">16.7% (10/60)</td>
                <td className="py-3 px-4 font-sans text-slate-400">
                  Literary prose descriptions diluted key lexical tokens and overwhelmed attention with adjective bias.
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-amber-300">4. Text-Only Inference (Transcript Only)</td>
                <td className="py-3 px-4 text-center">20.0% (6/30)</td>
                <td className="py-3 px-4 text-center">20.0% (6/30)</td>
                <td className="py-3 px-4 text-center text-amber-400">20.0% (12/60)</td>
                <td className="py-3 px-4 font-sans text-slate-400">
                  Without acoustic grounding, routine spoken disfluencies ("Just, just, just one addition...") were misclassified as frustration or anxiety.
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-amber-300">5. Static Weighted Score Fusion (0.6T + 0.4A)</td>
                <td className="py-3 px-4 text-center">50.0% (15/30)</td>
                <td className="py-3 px-4 text-center">36.7% (11/30)</td>
                <td className="py-3 px-4 text-center text-amber-400">43.3% (26/60)</td>
                <td className="py-3 px-4 font-sans text-slate-400">
                  Linear score averaging cannot perform non-linear XOR reasoning (Polite Words + Hostile Acoustics = Sarcasm).
                </td>
              </tr>
              <tr className="bg-emerald-950/20 font-semibold text-white">
                <td className="py-3 px-4 text-emerald-400">6. Vocalis Kev Tri-Modal Arbitration (Adopted)</td>
                <td className="py-3 px-4 text-center text-emerald-400">93.3% (28/30)</td>
                <td className="py-3 px-4 text-center text-emerald-400">66.7% (20/30)</td>
                <td className="py-3 px-4 text-center text-emerald-400 font-bold">80.0% (48/60)</td>
                <td className="py-3 px-4 font-sans text-emerald-300">
                  Block-causal PointerHead arbitrates verbatim transcript, Emotion2Vec+ prior, and Librosa prosody in a single calibrated forward pass.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Realistic Case Study Walkthrough */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-5">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
          <h2 className="text-base font-semibold text-white">
            Real Case Study: Acoustic Friction & Corporate Sarcasm Resolution
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-slate-400">
          How text-only models fail on polite corporate passive-aggression, and how Vocalis correctly arbitrates using multimodal acoustic context:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Signal 1: Lexical */}
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-xs font-mono text-slate-500 uppercase">Input 1: Lexical Words</span>
            <p className="text-sm font-serif italic text-slate-200">
              "Thank you so much for that very helpful clarification."
            </p>
            <div className="pt-2 border-t border-slate-800/80 text-xs">
              <span className="text-slate-400">Text-Only Sentiment:</span>
              <span className="text-emerald-400 font-medium ml-1.5">Positive (Delight)</span>
              <p className="text-[11px] text-rose-400 mt-1">❌ Misclassification Error</p>
            </div>
          </div>

          {/* Signal 2: Biomechanical Acoustics */}
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <span className="text-xs font-mono text-slate-500 uppercase">Input 2: Prosody & SER</span>
            <div className="space-y-1 font-mono text-xs text-slate-300">
              <div>
                Pitch: <span className="text-cyan-400">92 Hz (flat monotonic floor)</span>
              </div>
              <div>
                Tempo: <span className="text-amber-400">3.8 onsets/s (drawn out / tense)</span>
              </div>
              <div>
                Energy: <span className="text-rose-400">54 dB (suppressed dejected volume)</span>
              </div>
              <div>
                E2V+ Prior: <span className="text-purple-400">Disgust / Annoyance (91%)</span>
              </div>
            </div>
            <div className="pt-2 border-t border-slate-800/80 text-xs text-amber-300">
              ⚠️ Sharp Acoustic Divergence
            </div>
          </div>

          {/* Signal 3: Kev Multimodal Arbitration */}
          <div className="p-4 rounded-2xl bg-indigo-950/40 border border-indigo-500/40 space-y-2">
            <span className="text-xs font-mono text-indigo-400 uppercase">Output: Kev Decision</span>
            <div className="text-sm font-bold text-white flex items-center space-x-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Sarcasm / Passive-Aggression</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Kev synthesizes the tension: the caller uses polite phrasing cynically to conceal sharp vocal exasperation and contempt (<code className="text-indigo-300">noul = 0.89</code>).
            </p>
            <div className="pt-2 border-t border-indigo-800/60 text-xs font-mono text-emerald-400">
              ✅ 100% True Positive Detection
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
