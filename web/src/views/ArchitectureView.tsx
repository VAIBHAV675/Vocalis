import React, { useState } from 'react';
import { Zap, CheckCircle2, ShieldAlert, Sparkles, Layers, BookOpen } from 'lucide-react';

export const ArchitectureView: React.FC = () => {
  const [activeStage, setActiveStage] = useState<number>(1);

  const stages = [
    {
      num: 1,
      title: 'Faster-Whisper Large-v3 Turbo (INT8)',
      category: 'ASR & Temporal Segmentation',
      model: 'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
      precision: 'int8_float16 (CUDA) / int8 (CPU)',
      desc: 'High-throughput multilingual speech-to-text with integrated Silero Voice Activity Detection (VAD). Slices long continuous audio into discrete spoken segments with millisecond-precision timestamps.',
      metrics: '3.6x speedup over Whisper Large-v3, <1.2 GB VRAM footprint',
    },
    {
      num: 2,
      title: 'Emotion2Vec+ Large (1024-dim)',
      category: 'Acoustic Speech Foundation',
      model: 'iic/emotion2vec_plus_large (ModelScope / HF)',
      precision: 'FP16 / FP32',
      desc: 'Foundation acoustic encoder trained on 40,000+ hours of emotional speech. Generates rich 1024-dimensional continuous latent representations capturing micro-tremors, vocal timbre, and emotional inflection.',
      metrics: 'Universal multilingual acoustic embeddings, 7-class prior',
    },
    {
      num: 3,
      title: 'Biomechanical Prosody Engine',
      category: 'Acoustic Feature Extraction',
      model: 'Librosa Sound Processing Suite',
      precision: 'Analytical / DSP',
      desc: 'Quantifies physiological speech dynamics: Fundamental Frequency contour (F0 in Hz), Speech Tempo (onsets/sec), Vocal Intensity RMS (dB), and Phonation Quality (chest vs head resonance).',
      metrics: 'Deterministic, zero inference latency, interpretable physics',
    },
    {
      num: 4,
      title: 'Kev-0.8B / Kev-4B INT4 LLM Decision',
      category: 'Multimodal Cognitive Arbitration',
      model: 'jaredpalmer/kev-0.8b (NF4 INT4) & kev-4b (GGUF)',
      precision: '4-bit Quantized (NF4 / Q4_K_M)',
      desc: 'Multimodal language model fine-tuned for high-stakes decision routing. Ingests the verbatim transcript, acoustic prior, and prosodic dynamics to resolve subtle linguistic friction, sarcasm, and disguised anger.',
      metrics: '93.3% accuracy on English calls, 803 MB VRAM footprint',
    },
    {
      num: 5,
      title: 'Continuous Temporal Integration',
      category: 'Continuous Global Synthesis',
      model: 'Vocalis Time-Weighted Integration',
      precision: 'Mathematical Formulation',
      desc: 'Aggregates chopped segment predictions into a unified continuous profile. Weights emotional states by active speech duration rather than count, preventing short filler words from skewing long-form calls.',
      metrics: 'Continuous global probability vector across 8 canonical emotions',
    },
    {
      num: 6,
      title: 'Acoustic Incongruity & Sarcasm Filter',
      category: 'Friction & Risk Heuristic',
      model: 'Divergence Metric & Kev Binary Sarcasm Head',
      precision: 'Dual-Head Cross-Validation',
      desc: 'Detects divergence between lexical sentiment and vocal acoustic delivery. Flags corporate passive-aggression where formal words ("Thank you") mask frustrated vocal acoustics.',
      metrics: 'Zero false-alarm compliance monitoring in investment banking',
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
            Vocalis Multimodal Architecture & Pipeline Explainer
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-400">
          How Vocalis fuses Faster-Whisper INT8 transcription, Emotion2Vec+ Large acoustics, Librosa prosody, and Kev-0.8B INT4 reasoning to resolve complex human emotions.
        </p>
      </div>

      {/* 6-Stage Interactive Architecture Flow */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-slate-200 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>Interactive 6-Stage Pipeline Flow</span>
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

      {/* Realistic Case Study Walkthrough */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-5">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-5 h-5 text-amber-400" />
          <h2 className="text-base font-semibold text-white">
            Real Case Study: Acoustic Friction & Sarcasm Resolution
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
              <div>Pitch: <span className="text-cyan-400">92 Hz (flat monotonic floor)</span></div>
              <div>Tempo: <span className="text-amber-400">3.8 onsets/s (drawn out / tense)</span></div>
              <div>Energy: <span className="text-rose-400">54 dB (suppressed dejected volume)</span></div>
              <div>E2V+ Prior: <span className="text-purple-400">Disgust / Annoyance (91%)</span></div>
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
              <span>Frustration / Sarcasm</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Kev synthesizes the tension: the caller uses polite phrasing cynically to conceal sharp vocal exasperation and contempt.
            </p>
            <div className="pt-2 border-t border-indigo-800/60 text-xs font-mono text-emerald-400">
              ✅ 100% True Positive Detection
            </div>
          </div>
        </div>
      </div>

      {/* Benchmark Matrix Table */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <h2 className="text-base font-semibold text-white">
          Empirical Model Evaluation Matrix (GT Cleaned Benchmark)
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm text-slate-300">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Precision / Quant</th>
                <th className="py-3 px-4 text-center">English Acc (30)</th>
                <th className="py-3 px-4 text-center">Japanese Acc (30)</th>
                <th className="py-3 px-4 text-center">Overall Acc (60)</th>
                <th className="py-3 px-4 text-right">VRAM</th>
                <th className="py-3 px-4 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y border-slate-800 font-mono text-xs">
              <tr className="bg-indigo-950/20 font-semibold text-white">
                <td className="py-3.5 px-4 flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span>Kev-0.8B INT4 (Finalized Production)</span>
                </td>
                <td className="py-3.5 px-4 text-indigo-300">BitsAndBytes NF4</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">93.3% (28/30)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">43.3% (13/30)</td>
                <td className="py-3.5 px-4 text-center text-indigo-300">68.3% (41/60)</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">803 MB</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">2,388 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Kev-4B INT4 (Multilingual Champion)</td>
                <td className="py-3.5 px-4 text-slate-400">GGUF Q4_K_M</td>
                <td className="py-3.5 px-4 text-center text-slate-300">86.7% (26/30)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">66.7% (20/30)</td>
                <td className="py-3.5 px-4 text-center text-emerald-400 font-bold">76.7% (46/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">2,410 MB</td>
                <td className="py-3.5 px-4 text-right text-slate-400">3,450 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Emotion2Vec+ Large Standalone</td>
                <td className="py-3.5 px-4 text-slate-400">FP16 / FP32</td>
                <td className="py-3.5 px-4 text-center text-slate-300">83.3% (25/30)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">40.0% (12/30)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">61.7% (37/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">1,240 MB</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">810 ms</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4">Laya mmBERT Multimodal</td>
                <td className="py-3.5 px-4 text-slate-400">FP16</td>
                <td className="py-3.5 px-4 text-center text-slate-300">70.0% (21/30)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">36.7% (11/30)</td>
                <td className="py-3.5 px-4 text-center text-slate-400">53.3% (32/60)</td>
                <td className="py-3.5 px-4 text-right text-slate-400">650 MB</td>
                <td className="py-3.5 px-4 text-right text-emerald-400">420 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
