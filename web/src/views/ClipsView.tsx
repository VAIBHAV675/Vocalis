import React, { useState, useEffect, useMemo } from 'react';
import type { ClipItem, BenchmarkSummary } from '../types';
import { CANONICAL_EMOTIONS, SENTIMENT_META, emotionToSentiment } from '../types';
import {
  Play,
  Pause,
  Search,
  CheckCircle2,
  XCircle,
  ArrowUpRight,
  Layers,
  Zap,
  Gauge,
  ChevronLeft,
  ChevronRight,
  Award,
  Cpu,
} from 'lucide-react';

interface ClipsViewProps {
  onInspectClip: (clipId: string) => void;
}

const PAGE_SIZE = 48;

export const ClipsView: React.FC<ClipsViewProps> = ({ onInspectClip }) => {
  const [clips, setClips] = useState<ClipItem[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [benchmarkMode, setBenchmarkMode] = useState<'2000' | '60' | 'speed'>('2000');

  const [filterLang, setFilterLang] = useState<
    'all' | 'en' | 'ja' | 'audited60' | 'combined' | 'saved'
  >('all');
  const [filterEmotion, setFilterEmotion] = useState<string>('all');
  const [filterMatch, setFilterMatch] = useState<'all' | 'matched' | 'mismatch'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  const [playingClipId, setPlayingClipId] = useState<string | null>(null);
  const [audioObj, setAudioObj] = useState<HTMLAudioElement | null>(null);

  useEffect(() => {
    fetch('/api/clips')
      .then((res) => res.json())
      .then((data) => {
        setClips(data.clips || []);
        setSummary(data.benchmark_summary || null);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load clips:', err);
        setLoading(false);
      });

    return () => {
      if (audioObj) {
        audioObj.pause();
      }
    };
  }, []);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [filterLang, filterEmotion, filterMatch, searchQuery]);

  const togglePlayClip = (clip: ClipItem) => {
    if (playingClipId === clip.id) {
      if (audioObj) {
        audioObj.pause();
        setPlayingClipId(null);
      }
    } else {
      if (audioObj) {
        audioObj.pause();
      }
      const newAudio = new Audio(clip.audio_url);
      newAudio
        .play()
        .then(() => {
          setAudioObj(newAudio);
          setPlayingClipId(clip.id);
        })
        .catch(console.error);

      newAudio.onended = () => {
        setPlayingClipId(null);
      };
    }
  };

  const enCount = useMemo(
    () => clips.filter((c) => c.language === 'en' && c.type === 'single').length,
    [clips]
  );
  const jaCount = useMemo(
    () => clips.filter((c) => c.language === 'ja' && c.type === 'single').length,
    [clips]
  );
  const audited60Count = useMemo(
    () => clips.filter((c) => c.type === 'single' && c.is_audited_60).length,
    [clips]
  );
  const combinedCount = useMemo(
    () => clips.filter((c) => c.type === 'combined').length,
    [clips]
  );
  const savedCount = useMemo(
    () => clips.filter((c) => c.type === 'saved').length,
    [clips]
  );

  const filteredClips = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return clips.filter((c) => {
      const matchesLang =
        filterLang === 'all'
          ? true
          : filterLang === 'combined'
          ? c.type === 'combined'
          : filterLang === 'saved'
          ? c.type === 'saved'
          : filterLang === 'audited60'
          ? c.type === 'single' && Boolean(c.is_audited_60)
          : c.language === filterLang && c.type === 'single';

      const matchesEmotion =
        filterEmotion === 'all'
          ? true
          : c.predicted_emotion === filterEmotion || c.ground_truth_emotion === filterEmotion;

      const matchesAccuracy =
        filterMatch === 'all'
          ? true
          : filterMatch === 'matched'
          ? c.match
          : !c.match && c.type === 'single';

      const matchesSearch =
        !query ||
        c.id.toLowerCase().includes(query) ||
        c.title.toLowerCase().includes(query) ||
        c.ground_truth_transcript.toLowerCase().includes(query) ||
        c.domain.toLowerCase().includes(query);

      return matchesLang && matchesEmotion && matchesAccuracy && matchesSearch;
    });
  }, [clips, filterLang, filterEmotion, filterMatch, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filteredClips.length / PAGE_SIZE));
  const safePage = Math.min(currentPage, totalPages);
  const paginatedClips = useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return filteredClips.slice(start, start + PAGE_SIZE);
  }, [filteredClips, safePage]);

  return (
    <div className="space-y-6">
      {/* Title & Mode Switcher Banner */}
      <div className="p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 mb-1">
            <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
              <Layers className="w-4 h-4" />
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              {clips.length.toLocaleString()} Audio Clips Dataset & Speed/Accuracy Benchmarks
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-400">
            Complete catalog across 1,000 English Earnings22 calls, 1,000 Japanese ReazonSpeech broadcasts, 60 human-audited reference clips, and 2 continuous multi-minute evaluation streams.
          </p>
        </div>

        {/* Benchmark Telemetry View Switcher */}
        <div className="flex flex-wrap items-center gap-1.5 bg-slate-950/90 p-1.5 rounded-2xl border border-slate-800">
          <button
            onClick={() => {
              setBenchmarkMode('2000');
              if (filterLang === 'audited60') setFilterLang('all');
            }}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium transition flex items-center space-x-1.5 ${
              benchmarkMode === '2000'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>2,000-Clip Benchmark</span>
          </button>

          <button
            onClick={() => {
              setBenchmarkMode('60');
              setFilterLang('audited60');
            }}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium transition flex items-center space-x-1.5 ${
              benchmarkMode === '60'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Award className="w-3.5 h-3.5" />
            <span>60-Clip Audited Benchmark</span>
          </button>

          <button
            onClick={() => setBenchmarkMode('speed')}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium transition flex items-center space-x-1.5 ${
              benchmarkMode === 'speed'
                ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30'
                : 'text-emerald-400 hover:text-emerald-300'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Speed & Latency Benchmarks</span>
          </button>
        </div>
      </div>

      {/* Summary KPI Cards based on Selected Benchmark Mode */}
      {summary && benchmarkMode === '2000' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Total Audio Assets</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">{clips.length.toLocaleString()}</span>
              <span className="text-xs text-slate-500">Clips</span>
            </div>
            <span className="text-[11px] text-indigo-400 mt-1 block font-mono">
              1,000 EN + 1,000 JA + 2 Streams
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">1,000 English Calls Acc</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-emerald-400">
                {((summary.kev_4b_2000_en_acc || 0.835) * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">8-Emo (835/1K)</span>
            </div>
            <span className="text-[11px] text-emerald-300 mt-1 block font-mono">
              Sentiment Polarity: {((summary.english_sentiment_accuracy || 0.843) * 100).toFixed(1)}% (843/1K)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">1,000 Japanese Clips Acc</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-indigo-400">
                {((summary.kev_08b_2000_ja_acc || 0.473) * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">8-Emo (473/1K)</span>
            </div>
            <span className="text-[11px] text-indigo-300 mt-1 block font-mono">
              Sentiment Polarity: {((summary.japanese_sentiment_accuracy || 0.509) * 100).toFixed(1)}% (509/1K)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">2,000-Clip Multimodal Acc</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">
                {((summary.overall_accuracy || 0.654) * 100).toFixed(2)}%
              </span>
              <span className="text-xs text-slate-500">8-Emo</span>
            </div>
            <span className="text-[11px] text-emerald-400 mt-1 block font-mono">
              3-Way Sentiment: {((summary.overall_sentiment_accuracy || 0.676) * 100).toFixed(2)}% (1,352/2K)
            </span>
          </div>
        </div>
      )}

      {summary && benchmarkMode === '60' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="p-4 rounded-2xl bg-slate-900/90 border border-indigo-500/40 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Human-Audited Subset</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">60</span>
              <span className="text-xs text-slate-500">Clips + 2 Streams</span>
            </div>
            <span className="text-[11px] text-indigo-400 mt-1 block font-mono">
              30 EN (01–30) + 30 JA (01–30)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">English Audited Accuracy</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-emerald-400">93.3%</span>
              <span className="text-xs text-slate-500">8-Emo (28/30)</span>
            </div>
            <span className="text-[11px] text-emerald-300 mt-1 block font-mono">
              Sentiment Polarity: 96.7% (29/30)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Japanese Audited Accuracy</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-indigo-400">66.7%</span>
              <span className="text-xs text-slate-500">8-Emo (20/30)</span>
            </div>
            <span className="text-[11px] text-indigo-300 mt-1 block font-mono">
              Sentiment Polarity: 76.7% (23/30)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Language-Routed Overall</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">80.0%</span>
              <span className="text-xs text-slate-500">8-Emo (48/60)</span>
            </div>
            <span className="text-[11px] text-emerald-400 mt-1 block font-mono">
              3-Way Sentiment: 86.7% (52/60)
            </span>
          </div>
        </div>
      )}

      {summary && benchmarkMode === 'speed' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-emerald-500/40 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Faster-Whisper Turbo INT8</span>
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold text-emerald-400">2.8x–2.9x</span>
                <span className="text-xs text-slate-400">Real-Time</span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                EN: 2,175ms (0.35x RTF) • JA: 2,471ms
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Kev-0.8B C++ CUDA GGUF</span>
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold text-indigo-400">640 ms</span>
                <span className="text-xs text-slate-400">/ clip (1.51/s)</span>
              </div>
              <span className="text-[11px] text-emerald-400 mt-1 block font-mono">
                3.7x faster than PyTorch Triton (2,388ms)
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Emotion2Vec+ & Prosody DSP</span>
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold text-white">203 ms</span>
                <span className="text-xs text-slate-400">+ 58 ms DSP</span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block font-mono">
                1024-dim Acoustic Prior + YIN F0 / dB
              </span>
            </div>

            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Peak Active GPU VRAM</span>
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold text-emerald-400">&lt; 2.45 GB</span>
                <span className="text-xs text-slate-400">on 4GB GPU</span>
              </div>
              <span className="text-[11px] text-indigo-400 mt-1 block font-mono">
                Sequential Load → Infer → Evict
              </span>
            </div>
          </div>

          {/* Compact Speed & Latency Table */}
          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl overflow-x-auto">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-white flex items-center space-x-1.5">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span>Hardware Speed & Latency Benchmark Summary (NVIDIA GTX 1650 4GB VRAM)</span>
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                6.1s EN / 7.2s JA Reference Audio
              </span>
            </div>
            <table className="w-full text-left text-xs text-slate-300 font-mono">
              <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">Pipeline Stage / Model</th>
                  <th className="py-2 px-3">Precision / Backend</th>
                  <th className="py-2 px-3 text-right">Latency</th>
                  <th className="py-2 px-3 text-right">Throughput / RTF</th>
                  <th className="py-2 px-3 text-right">Active VRAM</th>
                  <th className="py-2 px-3">4GB Edge Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                <tr className="bg-emerald-950/15">
                  <td className="py-2 px-3 font-semibold text-white">Faster-Whisper Large-v3 Turbo (EN / JA)</td>
                  <td className="py-2 px-3 text-indigo-300">CTranslate2 int8_float16</td>
                  <td className="py-2 px-3 text-right text-emerald-400">2,175 ms / 2,471 ms</td>
                  <td className="py-2 px-3 text-right text-emerald-400">0.35x RTF (2.8x–2.9x Realtime)</td>
                  <td className="py-2 px-3 text-right">1,120 MB</td>
                  <td className="py-2 px-3 text-emerald-400">✅ Selected Production ASR</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Faster-Whisper Large-v3 Turbo (FP16)</td>
                  <td className="py-2 px-3 text-slate-500">float16</td>
                  <td className="py-2 px-3 text-right text-rose-400">—</td>
                  <td className="py-2 px-3 text-right text-rose-400">—</td>
                  <td className="py-2 px-3 text-right text-rose-400">&gt; 4,096 MB</td>
                  <td className="py-2 px-3 text-rose-400">❌ CUDA OOM on 4GB GPU</td>
                </tr>
                <tr>
                  <td className="py-2 px-3 text-slate-400">Kotoba-Whisper v2.0 Distilled (JA)</td>
                  <td className="py-2 px-3 text-slate-400">float16 / int8_float16</td>
                  <td className="py-2 px-3 text-right">8,699 ms / 2,227 ms</td>
                  <td className="py-2 px-3 text-right">1.43x / 0.37x RTF</td>
                  <td className="py-2 px-3 text-right">1,650 MB</td>
                  <td className="py-2 px-3 text-amber-400">⚠️ Hallucination loops + 2nd load</td>
                </tr>
                <tr>
                  <td className="py-2 px-3">Emotion2Vec+ Large Foundation</td>
                  <td className="py-2 px-3 text-slate-400">FunASR FP16</td>
                  <td className="py-2 px-3 text-right text-emerald-400">203 ms</td>
                  <td className="py-2 px-3 text-right text-emerald-400">~30x Realtime</td>
                  <td className="py-2 px-3 text-right">1,240 MB</td>
                  <td className="py-2 px-3 text-emerald-400">✅ Stage 2 Acoustic Prior</td>
                </tr>
                <tr>
                  <td className="py-2 px-3">Librosa Biomechanical Prosody DSP</td>
                  <td className="py-2 px-3 text-slate-400">CPU Vectorized</td>
                  <td className="py-2 px-3 text-right text-emerald-400">58 ms</td>
                  <td className="py-2 px-3 text-right text-emerald-400">~105x Realtime</td>
                  <td className="py-2 px-3 text-right">0 MB (CPU)</td>
                  <td className="py-2 px-3 text-emerald-400">✅ Stage 3 Prosody Engine</td>
                </tr>
                <tr className="bg-indigo-950/20">
                  <td className="py-2 px-3 font-semibold text-white">Kev-0.8B Multimodal Arbitration</td>
                  <td className="py-2 px-3 text-indigo-300">GGUF Q8_0 (C++ CUDA)</td>
                  <td className="py-2 px-3 text-right text-emerald-400">545 ms – 640 ms</td>
                  <td className="py-2 px-3 text-right text-emerald-400">1.51 clips/s (3.7x vs PyTorch)</td>
                  <td className="py-2 px-3 text-right text-emerald-400">803 MB</td>
                  <td className="py-2 px-3 text-emerald-400">✅ Fast Stage 4 Decision Head</td>
                </tr>
                <tr className="bg-indigo-950/10">
                  <td className="py-2 px-3 font-semibold text-white">Kev-4B Multimodal Arbitration</td>
                  <td className="py-2 px-3 text-indigo-300">GGUF Q4_K_M (C++ CUDA)</td>
                  <td className="py-2 px-3 text-right text-slate-300">2,853 ms – 3,154 ms</td>
                  <td className="py-2 px-3 text-right text-slate-300">0.31 clips/s</td>
                  <td className="py-2 px-3 text-right">2,410 MB</td>
                  <td className="py-2 px-3 text-emerald-400">✅ High-Capacity INT4 Head</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="flex flex-col xl:flex-row items-stretch xl:items-center justify-between gap-3 p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
        {/* Language & Category Tabs */}
        <div className="flex flex-wrap items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setFilterLang('all')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'all'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({clips.length.toLocaleString()})
          </button>
          <button
            onClick={() => setFilterLang('en')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'en'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            English Calls ({enCount.toLocaleString()})
          </button>
          <button
            onClick={() => setFilterLang('ja')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'ja'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Japanese ({jaCount.toLocaleString()})
          </button>
          <button
            onClick={() => setFilterLang('audited60')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'audited60'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-indigo-300 hover:text-white'
            }`}
          >
            60 Audited Subset ({audited60Count})
          </button>
          <button
            onClick={() => setFilterLang('combined')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'combined'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Continuous ({combinedCount})
          </button>
          {savedCount > 0 && (
            <button
              onClick={() => setFilterLang('saved')}
              className={`px-3 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 ${
                filterLang === 'saved'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10'
              }`}
            >
              <span>Saved Recordings</span>
              <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-emerald-500/30 text-emerald-200 font-mono">
                {savedCount}
              </span>
            </button>
          )}
        </div>

        {/* Emotion Filter, Accuracy Filter & Search */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div className="relative flex-1 sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search transcript, ID (e.g. en_ib_sample_450)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 text-xs"
            />
          </div>

          <select
            value={filterMatch}
            onChange={(e) => setFilterMatch(e.target.value as 'all' | 'matched' | 'mismatch')}
            className="px-2.5 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 focus:outline-none focus:border-indigo-500 text-xs"
          >
            <option value="all">All Predictions</option>
            <option value="matched">✅ Matched Ground Truth</option>
            <option value="mismatch">⚠️ Disagreed with GT</option>
          </select>

          <select
            value={filterEmotion}
            onChange={(e) => setFilterEmotion(e.target.value)}
            className="px-2.5 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 focus:outline-none focus:border-indigo-500 text-xs"
          >
            <option value="all">All Emotions</option>
            {Object.entries(CANONICAL_EMOTIONS).map(([k, m]) => (
              <option key={k} value={k}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Pagination & Result Count Bar */}
      {!loading && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 px-1 text-xs text-slate-400">
          <div>
            Showing{' '}
            <span className="font-mono font-semibold text-white">
              {filteredClips.length === 0 ? 0 : (safePage - 1) * PAGE_SIZE + 1}–
              {Math.min(safePage * PAGE_SIZE, filteredClips.length)}
            </span>{' '}
            of{' '}
            <span className="font-mono font-semibold text-indigo-400">
              {filteredClips.length.toLocaleString()}
            </span>{' '}
            matching audio clips
          </div>

          {totalPages > 1 && (
            <div className="flex items-center space-x-1.5">
              <button
                disabled={safePage <= 1}
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="font-mono text-xs px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-200">
                Page {safePage} / {totalPages}
              </span>
              <button
                disabled={safePage >= totalPages}
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Clips Cards Grid */}
      {loading ? (
        <div className="text-center py-20 text-slate-400">
          Loading 2,000+ audio clip catalog & benchmark telemetry...
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {paginatedClips.map((clip) => {
            const isPlaying = playingClipId === clip.id;
            const emotionMeta = CANONICAL_EMOTIONS[
              clip.predicted_emotion as keyof typeof CANONICAL_EMOTIONS
            ] || {
              label: clip.predicted_emotion,
              color: '#64748b',
              badgeBg: 'bg-slate-800 text-slate-300 border-slate-700',
            };
            const gtMeta = CANONICAL_EMOTIONS[
              clip.ground_truth_emotion as keyof typeof CANONICAL_EMOTIONS
            ];

            return (
              <div
                key={clip.id}
                className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800/90 hover:border-slate-700 hover:shadow-xl transition flex flex-col justify-between space-y-3"
              >
                {/* Header */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => togglePlayClip(clip)}
                      className={`w-8 h-8 rounded-xl flex items-center justify-center transition flex-shrink-0 ${
                        isPlaying
                          ? 'bg-indigo-600 text-white shadow-md animate-pulse'
                          : 'bg-slate-800 text-slate-300 hover:bg-indigo-600 hover:text-white'
                      }`}
                    >
                      {isPlaying ? (
                        <Pause className="w-3.5 h-3.5 fill-current" />
                      ) : (
                        <Play className="w-3.5 h-3.5 fill-current ml-0.5" />
                      )}
                    </button>
                    <div>
                      <div className="flex items-center flex-wrap gap-1.5">
                        <span className="font-semibold text-xs text-white truncate max-w-[165px]">
                          {clip.title}
                        </span>
                        <span
                          className={`text-[10px] uppercase px-1.5 py-0.2 rounded font-mono ${
                            clip.type === 'saved'
                              ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-300'
                              : 'bg-slate-950 border border-slate-800 text-indigo-400'
                          }`}
                        >
                          {clip.type === 'saved' ? 'SAVED' : clip.language}
                        </span>
                        {clip.type === 'single' && clip.is_audited_60 && (
                          <span className="text-[9px] uppercase px-1.5 py-0.2 rounded font-mono bg-amber-500/15 border border-amber-500/30 text-amber-300">
                            GT-60
                          </span>
                        )}
                      </div>
                      <span className="text-[11px] text-slate-500 font-mono block">
                        {clip.duration_sec.toFixed(1)}s •{' '}
                        {clip.type === 'saved' ? 'Recorded speech' : clip.domain.split('/')[0]}
                        {clip.latency_ms ? ` • ${Math.round(clip.latency_ms)}ms` : ''}
                      </span>
                    </div>
                  </div>

                  {/* Match status */}
                  {clip.type === 'saved' ? (
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                      User Audio
                    </span>
                  ) : (
                    clip.type !== 'combined' && (
                      <span
                        title={
                          clip.match
                            ? 'Prediction matched Ground Truth'
                            : `Ground Truth: ${clip.ground_truth_emotion}`
                        }
                        className="flex items-center"
                      >
                        {clip.match ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <XCircle className="w-4 h-4 text-amber-400" />
                        )}
                      </span>
                    )
                  )}
                </div>

                {/* Transcript */}
                <p className="text-xs text-slate-300 line-clamp-3 italic leading-relaxed">
                  "{clip.ground_truth_transcript}"
                </p>

                {/* Prosody & Model Micro-Telemetry */}
                {clip.type === 'single' && (
                  <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-slate-400">
                    {gtMeta && (
                      <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                        GT: {gtMeta.label.split('/')[0].trim()}
                      </span>
                    )}
                    {clip.ser_prior && (
                      <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-indigo-300">
                        E2V+: {clip.ser_prior}
                      </span>
                    )}
                    {clip.prosody_summary?.pitch && (
                      <span className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 text-cyan-300 flex items-center space-x-1">
                        <Gauge className="w-2.5 h-2.5" />
                        <span>{clip.prosody_summary.pitch.split('(')[0].trim()}</span>
                      </span>
                    )}
                  </div>
                )}

                {/* Sentiment Polarity & Emotion Badges & Inspect Button */}
                <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {(() => {
                      const sk = clip.predicted_sentiment || emotionToSentiment(clip.predicted_emotion);
                      const sm = SENTIMENT_META[sk] || SENTIMENT_META.neutral;
                      return (
                        <span
                          className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-semibold border flex items-center space-x-1 ${sm.badgeBg}`}
                        >
                          <span
                            className="w-1.5 h-1.5 rounded-full"
                            style={{ backgroundColor: sm.color }}
                          />
                          <span>{sm.shortLabel}</span>
                        </span>
                      );
                    })()}
                    <span
                      className={`text-[11px] px-2 py-0.5 rounded-full font-medium border flex items-center space-x-1 ${emotionMeta.badgeBg}`}
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: emotionMeta.color }}
                      />
                      <span>{emotionMeta.label.split('/')[0]}</span>
                    </span>
                  </div>

                  <button
                    onClick={() => onInspectClip(clip.id)}
                    className="flex items-center space-x-1 text-[11px] font-medium text-indigo-400 hover:text-indigo-300 transition whitespace-nowrap"
                  >
                    <span>Inspect Pipeline</span>
                    <ArrowUpRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Bottom Pagination Controls */}
      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-center space-x-2 pt-4">
          <button
            disabled={safePage <= 1}
            onClick={() => {
              setCurrentPage((p) => Math.max(1, p - 1));
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition flex items-center space-x-1"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous</span>
          </button>

          <span className="font-mono text-xs px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300">
            Page {safePage} of {totalPages} ({filteredClips.length.toLocaleString()} clips)
          </span>

          <button
            disabled={safePage >= totalPages}
            onClick={() => {
              setCurrentPage((p) => Math.min(totalPages, p + 1));
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition flex items-center space-x-1"
          >
            <span>Next</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
