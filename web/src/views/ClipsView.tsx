import React, { useState, useEffect } from 'react';
import type { ClipItem, BenchmarkSummary } from '../types';
import { CANONICAL_EMOTIONS } from '../types';
import { Play, Pause, Search, CheckCircle2, XCircle, ArrowUpRight, Layers } from 'lucide-react';

interface ClipsViewProps {
  onInspectClip: (clipId: string) => void;
}

export const ClipsView: React.FC<ClipsViewProps> = ({ onInspectClip }) => {
  const [clips, setClips] = useState<ClipItem[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterLang, setFilterLang] = useState<'all' | 'en' | 'ja' | 'combined' | 'saved'>('all');
  const [filterEmotion, setFilterEmotion] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
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
      newAudio.play().then(() => {
        setAudioObj(newAudio);
        setPlayingClipId(clip.id);
      }).catch(console.error);

      newAudio.onended = () => {
        setPlayingClipId(null);
      };
    }
  };

  const enCount = clips.filter((c) => c.language === 'en' && c.type !== 'combined' && c.type !== 'saved').length;
  const jaCount = clips.filter((c) => c.language === 'ja' && c.type !== 'combined' && c.type !== 'saved').length;
  const combinedCount = clips.filter((c) => c.type === 'combined').length;
  const savedCount = clips.filter((c) => c.type === 'saved').length;

  const filteredClips = clips.filter((c) => {
    const matchesLang =
      filterLang === 'all'
        ? true
        : filterLang === 'combined'
        ? c.type === 'combined'
        : filterLang === 'saved'
        ? c.type === 'saved'
        : c.language === filterLang && c.type !== 'combined' && c.type !== 'saved';

    const matchesEmotion =
      filterEmotion === 'all'
        ? true
        : c.predicted_emotion === filterEmotion || c.ground_truth_emotion === filterEmotion;

    const query = searchQuery.toLowerCase();
    const matchesSearch =
      c.id.toLowerCase().includes(query) ||
      c.ground_truth_transcript.toLowerCase().includes(query) ||
      c.domain.toLowerCase().includes(query);

    return matchesLang && matchesEmotion && matchesSearch;
  });

  return (
    <div className="space-y-6">
      {/* Title Banner */}
      <div className="p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl">
        <div className="flex items-center space-x-2 mb-1">
          <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Layers className="w-4 h-4" />
          </span>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            62 Audio Clips Dataset & Benchmark History
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-400">
          Comprehensive catalog across 30 English Earnings22 calls, 30 Japanese ReazonSpeech broadcasts, and 2 continuous concatenated evaluation streams.
        </p>
      </div>

      {/* Summary KPI Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Total Audio Assets</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">{clips.length}</span>
              <span className="text-xs text-slate-500">Clips</span>
            </div>
            <span className="text-[11px] text-indigo-400 mt-1 block font-mono">
              30 EN + 30 JA + 2 Combined
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">English Call Accuracy</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-emerald-400">
                {(summary.english_accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">28 / 30 correct</span>
            </div>
            <span className="text-[11px] text-slate-400 mt-1 block">
              Kev-0.8B INT4 (BitsAndBytes NF4)
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Japanese Accuracy</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-indigo-400">
                {(summary.japanese_accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">20 / 30 correct</span>
            </div>
            <span className="text-[11px] text-slate-400 mt-1 block">
              Kev-4B INT4 Multilingual
            </span>
          </div>

          <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 block mb-1">Overall Multimodal Acc</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold text-white">
                {(summary.overall_accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">46 / 60 correct</span>
            </div>
            <span className="text-[11px] text-slate-400 mt-1 block">
              Cleaned Ground Truth
            </span>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
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
            All ({clips.length})
          </button>
          <button
            onClick={() => setFilterLang('en')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'en'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            English Calls ({enCount})
          </button>
          <button
            onClick={() => setFilterLang('ja')}
            className={`px-3 py-1.5 rounded-lg font-medium transition ${
              filterLang === 'ja'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Japanese ({jaCount})
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

        {/* Emotion Filter & Search */}
        <div className="flex items-center space-x-2 text-xs">
          <div className="relative flex-1 sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search transcript, ID, domain..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 text-xs"
            />
          </div>

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

      {/* Clips Cards Grid */}
      {loading ? (
        <div className="text-center py-20 text-slate-400">Loading audio catalog...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredClips.map((clip) => {
            const isPlaying = playingClipId === clip.id;
            const emotionMeta = CANONICAL_EMOTIONS[clip.predicted_emotion as keyof typeof CANONICAL_EMOTIONS] || {
              label: clip.predicted_emotion,
              color: '#64748b',
              badgeBg: 'bg-slate-800 text-slate-300 border-slate-700',
            };

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
                      className={`w-8 h-8 rounded-xl flex items-center justify-center transition ${
                        isPlaying
                          ? 'bg-indigo-600 text-white shadow-md animate-pulse'
                          : 'bg-slate-800 text-slate-300 hover:bg-indigo-600 hover:text-white'
                      }`}
                    >
                      {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current ml-0.5" />}
                    </button>
                    <div>
                      <div className="flex items-center space-x-1.5">
                        <span className="font-semibold text-xs text-white truncate max-w-[170px]">
                          {clip.title}
                        </span>
                        <span className={`text-[10px] uppercase px-1.5 py-0.2 rounded font-mono ${
                          clip.type === 'saved'
                            ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-300'
                            : 'bg-slate-950 border border-slate-800 text-indigo-400'
                        }`}>
                          {clip.type === 'saved' ? 'SAVED' : clip.language}
                        </span>
                      </div>
                      <span className="text-[11px] text-slate-500 font-mono">
                        {clip.duration_sec.toFixed(1)}s • {clip.type === 'saved' ? `Recorded speech` : clip.domain.split('/')[0]}
                      </span>
                    </div>
                  </div>

                  {/* Match status */}
                  {clip.type === 'saved' ? (
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                      User Audio
                    </span>
                  ) : clip.type !== 'combined' && (
                    <span
                      title={clip.match ? 'Prediction matched Ground Truth' : 'Prediction differed from Ground Truth'}
                      className="flex items-center"
                    >
                      {clip.match ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <XCircle className="w-4 h-4 text-amber-400" />
                      )}
                    </span>
                  )}
                </div>

                {/* Transcript */}
                <p className="text-xs text-slate-300 line-clamp-3 italic leading-relaxed">
                  "{clip.ground_truth_transcript}"
                </p>

                {/* Emotion Badges & Inspect Button */}
                <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-1.5">
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
                    className="flex items-center space-x-1 text-[11px] font-medium text-indigo-400 hover:text-indigo-300 transition"
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
    </div>
  );
};
