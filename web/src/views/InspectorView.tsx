import React, { useState, useEffect, useRef } from 'react';
import { WaveformPlayer } from '../components/WaveformPlayer';
import type { WaveformPlayerHandle } from '../components/WaveformPlayer';
import { EmotionBarChart } from '../components/EmotionBarChart';
import { SegmentCard } from '../components/SegmentCard';
import type { PipelineResult, Segment, SavedRecordingItem } from '../types';
import { CANONICAL_EMOTIONS, SENTIMENT_META, emotionToSentiment, computeSentimentProbabilities } from '../types';
import {
  Sparkles,
  Filter,
  Search,
  Globe,
  ShieldAlert,
  Layers,
  FolderHeart,
  CheckCircle2,
  XCircle,
  Zap,
  AlertTriangle,
  ArrowRight,
} from 'lucide-react';

interface InspectorViewProps {
  customResult?: PipelineResult | null;
  onSelectClipToInspect?: (clipId: string) => void;
}

export const InspectorView: React.FC<InspectorViewProps> = ({ customResult }) => {
  const [selectedStream, setSelectedStream] = useState<'en' | 'ja' | 'custom'>(() =>
    customResult ? 'custom' : 'en'
  );
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pipelineData, setPipelineData] = useState<PipelineResult | null>(customResult || null);
  const [activeSegment, setActiveSegment] = useState<Segment | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [emotionFilter, setEmotionFilter] = useState<string>('all');
  const [savedRecordings, setSavedRecordings] = useState<SavedRecordingItem[]>([]);

  // Quick 2,000-Clip Jumper state
  const [quickLang, setQuickLang] = useState<'en' | 'ja'>('en');
  const [quickNum, setQuickNum] = useState<string>('1');

  const playerRef = useRef<WaveformPlayerHandle | null>(null);

  // Load saved recordings on mount
  useEffect(() => {
    fetch('/api/saved_recordings')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setSavedRecordings(data);
        }
      })
      .catch((err) => console.error('Failed to load saved recordings:', err));
  }, []);

  // When customResult changes from outside (e.g. clicking "Inspect Pipeline" in ClipsView), switch to custom view immediately
  useEffect(() => {
    if (customResult) {
      setSelectedStream('custom');
      setPipelineData(customResult);
      setErrorMsg(null);
      setActiveSegment(null);
    }
  }, [customResult]);

  // Load pipeline data when stream changes
  useEffect(() => {
    if (selectedStream === 'custom') {
      if (customResult && !pipelineData) {
        setPipelineData(customResult);
      }
      return;
    }

    const clipId =
      selectedStream === 'ja' ? 'combined_japanese_30_broadcasts' : 'combined_english_30_calls';
    setLoading(true);
    setErrorMsg(null);

    fetch(`/api/pipeline/${clipId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load pipeline results`);
        return res.json();
      })
      .then((data: PipelineResult) => {
        setPipelineData(data);
        setActiveSegment(null);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setErrorMsg(err.message || 'Failed to load pipeline telemetry.');
        setLoading(false);
      });
  }, [selectedStream]);

  // Load a specific clip or saved recording into Inspector
  const handleLoadClipById = async (clipId: string) => {
    if (!clipId) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/pipeline/${clipId}`);
      if (!res.ok) {
        throw new Error(`Clip '${clipId}' not found (HTTP ${res.status})`);
      }
      const data: PipelineResult = await res.json();
      setSelectedStream('custom');
      setPipelineData(data);
      setActiveSegment(null);
    } catch (err: any) {
      console.error('Failed to load clip for inspection:', err);
      setErrorMsg(err.message || 'Failed to load clip.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickJump = (e: React.FormEvent) => {
    e.preventDefault();
    const n = Math.max(1, Math.min(1000, parseInt(quickNum, 10) || 1));
    setQuickNum(String(n));
    const padded = n < 100 ? String(n).padStart(2, '0') : String(n);
    const clipId = `${quickLang}_ib_sample_${padded}`;
    handleLoadClipById(clipId);
  };

  // Sync active segment when currentTime changes
  const handleTimeUpdate = (time: number) => {
    if (!pipelineData) return;

    const currentSeg = pipelineData.segments.find(
      (s) => time >= s.start_sec && time <= s.end_sec
    );
    setActiveSegment(currentSeg || null);
  };

  // Play specific segment when clicked
  const handlePlaySegment = (segment: Segment) => {
    if (playerRef.current) {
      playerRef.current.seekTo(segment.start_sec, true, segment.end_sec);
      setActiveSegment(segment);
    }
  };

  // Filter segments
  const filteredSegments = (pipelineData?.segments || []).filter((seg) => {
    const matchesSearch = seg.transcript.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesEmotion = emotionFilter === 'all' || seg.predicted_emotion === emotionFilter;
    return matchesSearch && matchesEmotion;
  });

  const dominantMeta = pipelineData
    ? CANONICAL_EMOTIONS[pipelineData.dominant_emotion] || CANONICAL_EMOTIONS.neutral_inquiry
    : null;

  const dominantSentKey = pipelineData
    ? pipelineData.dominant_sentiment ||
      emotionToSentiment(pipelineData.dominant_emotion, pipelineData.sarcastic_segments_count > 0 && pipelineData.total_segments === 1)
    : 'neutral';
  const dominantSentMeta = SENTIMENT_META[dominantSentKey] || SENTIMENT_META.neutral;
  const globalSentProbs = pipelineData
    ? pipelineData.global_sentiment_probability_vector ||
      computeSentimentProbabilities(
        pipelineData.global_probability_vector as unknown as Record<string, number>,
        pipelineData.segments[0]?.sarcasm_probability,
        pipelineData.sarcastic_segments_count > 0 && pipelineData.total_segments === 1
      )
    : { positive: 0, neutral: 1, negative: 0 };

  const gtMeta =
    pipelineData?.ground_truth_emotion &&
    CANONICAL_EMOTIONS[pipelineData.ground_truth_emotion as keyof typeof CANONICAL_EMOTIONS]
      ? CANONICAL_EMOTIONS[pipelineData.ground_truth_emotion as keyof typeof CANONICAL_EMOTIONS]
      : null;

  return (
    <div className="space-y-6">
      {/* Stream Selector & Overview Banner */}
      <div className="flex flex-col gap-4 p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 mb-1">
              <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
                <Sparkles className="w-4 h-4" />
              </span>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                Multimodal Pipeline Inspector
              </h1>
            </div>
            <p className="text-xs sm:text-sm text-slate-400">
              Inspect continuous multi-minute streams or jump directly to any of the 2,000 benchmark clips (`1,000 EN` + `1,000 JA`). Click any segment card to seek the waveform.
            </p>
          </div>

          {/* Stream Pills */}
          <div className="flex flex-wrap items-center gap-2 bg-slate-950/90 p-1.5 rounded-2xl border border-slate-800/80">
            <button
              onClick={() => setSelectedStream('en')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition ${
                selectedStream === 'en'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Globe className="w-4 h-4" />
              <span>Continuous English (30 Calls)</span>
            </button>

            <button
              onClick={() => setSelectedStream('ja')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition ${
                selectedStream === 'ja'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Globe className="w-4 h-4" />
              <span>Continuous Japanese (30 Clips)</span>
            </button>

            {selectedStream === 'custom' && pipelineData && (
              <button
                onClick={() => setSelectedStream('custom')}
                className="flex items-center space-x-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 transition"
              >
                <Sparkles className="w-4 h-4 text-emerald-300" />
                <span className="truncate max-w-[180px]">
                  {pipelineData.clip_id || pipelineData.id || 'Inspected Clip'}
                </span>
              </button>
            )}
          </div>
        </div>

        {/* Quick 2,000-Clip Jumper & Saved Recordings Bar */}
        <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3 text-xs">
          <form onSubmit={handleQuickJump} className="flex flex-wrap items-center gap-2">
            <span className="text-slate-400 font-medium">Quick Inspect 2,000-Clip Dataset:</span>
            <select
              value={quickLang}
              onChange={(e) => setQuickLang(e.target.value as 'en' | 'ja')}
              className="bg-slate-950 border border-slate-800 text-slate-200 px-2.5 py-1.5 rounded-xl focus:outline-none focus:border-indigo-500 font-mono"
            >
              <option value="en">English (en_ib_sample_01 – 1000)</option>
              <option value="ja">Japanese (ja_ib_sample_01 – 1000)</option>
            </select>
            <div className="flex items-center space-x-1">
              <span className="text-slate-500 font-mono">#</span>
              <input
                type="number"
                min={1}
                max={1000}
                value={quickNum}
                onChange={(e) => setQuickNum(e.target.value)}
                className="w-20 bg-slate-950 border border-slate-800 text-slate-200 px-2.5 py-1.5 rounded-xl focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
            <button
              type="submit"
              className="px-3 py-1.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/30 font-medium transition flex items-center space-x-1"
            >
              <span>Load Clip</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </form>

          {savedRecordings.length > 0 && (
            <div className="flex items-center space-x-2">
              <FolderHeart className="w-3.5 h-3.5 text-pink-400" />
              <select
                value={
                  selectedStream === 'custom' && pipelineData?.id?.startsWith('rec_')
                    ? pipelineData.id
                    : ''
                }
                onChange={(e) => handleLoadClipById(e.target.value)}
                className="bg-slate-950 border border-slate-800 text-xs text-slate-200 px-2.5 py-1.5 rounded-xl focus:outline-none focus:border-indigo-500 font-medium"
              >
                <option value="" disabled>
                  Saved Recordings ({savedRecordings.length})...
                </option>
                {savedRecordings.map((rec) => (
                  <option key={rec.id} value={rec.id}>
                    {rec.title} ({rec.dominant_emotion.replace('_', ' ')})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-20 text-slate-400 space-x-3">
          <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span>Loading multimodal pipeline timeline...</span>
        </div>
      )}

      {errorMsg && !loading && (
        <div className="p-6 rounded-2xl bg-rose-950/30 border border-rose-500/40 text-rose-200 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-semibold">Failed to load pipeline telemetry</h3>
              <p className="text-xs text-rose-300/80 mt-0.5">{errorMsg}</p>
            </div>
          </div>
          <button
            onClick={() => setSelectedStream('en')}
            className="px-3 py-1.5 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-xs font-medium border border-rose-500/30 transition"
          >
            Reset to English Stream
          </button>
        </div>
      )}

      {!loading && !errorMsg && pipelineData && (
        <>
          {/* Audio Waveform Player */}
          <WaveformPlayer
            ref={playerRef}
            audioUrl={pipelineData.audio_url || ''}
            title={
              pipelineData.title ||
              `Continuous Multimodal Stream (${pipelineData.language.toUpperCase()})`
            }
            onTimeUpdate={handleTimeUpdate}
            activeSegmentRange={
              activeSegment ? { start: activeSegment.start_sec, end: activeSegment.end_sec } : null
            }
          />

          {/* Single-Clip Benchmark Telemetry Comparison (shown when inspecting one of the 2,000 dataset clips) */}
          {pipelineData.ground_truth_emotion && (
            <div className="p-4 rounded-2xl bg-indigo-950/25 border border-indigo-500/30 flex flex-wrap items-center justify-between gap-4 text-xs">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <span className="text-slate-400 block text-[11px]">Audited Ground Truth</span>
                  <span className="font-semibold text-white flex items-center space-x-1.5 mt-0.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: gtMeta?.color || '#94a3b8' }}
                    />
                    <span>{gtMeta?.label || pipelineData.ground_truth_emotion}</span>
                  </span>
                </div>

                <div className="h-7 w-px bg-slate-800 hidden sm:block" />

                {pipelineData.kev_08b_prediction && (
                  <div>
                    <span className="text-slate-400 block text-[11px]">Kev-0.8B (GGUF Q8_0)</span>
                    <span className="font-mono text-slate-200 flex items-center space-x-1 mt-0.5">
                      {pipelineData.kev_08b_prediction === pipelineData.ground_truth_emotion ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-amber-400" />
                      )}
                      <span>{pipelineData.kev_08b_prediction}</span>
                    </span>
                  </div>
                )}

                {pipelineData.kev_4b_prediction && (
                  <div>
                    <span className="text-slate-400 block text-[11px]">Kev-4B (GGUF Q4_K_M)</span>
                    <span className="font-mono text-slate-200 flex items-center space-x-1 mt-0.5">
                      {pipelineData.kev_4b_prediction === pipelineData.ground_truth_emotion ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-amber-400" />
                      )}
                      <span>{pipelineData.kev_4b_prediction}</span>
                    </span>
                  </div>
                )}
              </div>

              {pipelineData.latency_ms && (
                <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-950/90 border border-slate-800 font-mono text-emerald-400">
                  <Zap className="w-3.5 h-3.5" />
                  <span>Inference Latency: {pipelineData.latency_ms.toFixed(0)} ms</span>
                </div>
              )}
            </div>
          )}

          {/* Top Key Metrics Banner */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 sm:gap-4">
            {/* Dominant 3-Way Sentiment */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Overall Sentiment Polarity</span>
              <div className="flex items-center space-x-2">
                <span
                  className="w-3 h-3 rounded-full shadow-sm"
                  style={{ backgroundColor: dominantSentMeta.color }}
                />
                <span className="text-base sm:text-lg font-bold text-white truncate">
                  {dominantSentMeta.shortLabel}
                </span>
              </div>
              <span className="text-[11px] text-slate-400 font-mono mt-1 block">
                +{Math.round((globalSentProbs.positive || 0) * 100)}% / ~{Math.round((globalSentProbs.neutral || 0) * 100)}% / -{Math.round((globalSentProbs.negative || 0) * 100)}%
              </span>
            </div>

            {/* Dominant Emotion */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Dominant Overall Emotion</span>
              <div className="flex items-center space-x-2">
                <span
                  className="w-3 h-3 rounded-full shadow-sm"
                  style={{ backgroundColor: dominantMeta?.color }}
                />
                <span className="text-base sm:text-lg font-bold text-white truncate">
                  {dominantMeta?.label.split('/')[0]}
                </span>
              </div>
              <span className="text-xs text-indigo-400 font-mono mt-1 block">
                {Math.round(
                  pipelineData.dominant_probability <= 1
                    ? pipelineData.dominant_probability * 100
                    : pipelineData.dominant_probability
                )}
                % probability
              </span>
            </div>

            {/* Time Share */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Dominant Time-Share</span>
              <div className="flex items-baseline space-x-1.5">
                <span className="text-xl sm:text-2xl font-bold text-white">
                  {(
                    pipelineData.time_distribution_percent[pipelineData.dominant_emotion] || 0
                  ).toFixed(1)}
                  %
                </span>
                <span className="text-xs text-slate-500">of spoken audio</span>
              </div>
              <span className="text-xs text-slate-400 font-mono mt-1 block">
                {pipelineData.spoken_duration_sec.toFixed(1)}s /{' '}
                {pipelineData.total_duration_sec.toFixed(1)}s
              </span>
            </div>

            {/* Segments Count */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Extracted Segments</span>
              <div className="flex items-center space-x-2">
                <Layers className="w-5 h-5 text-indigo-400" />
                <span className="text-xl sm:text-2xl font-bold text-white">
                  {pipelineData.total_segments}
                </span>
              </div>
              <span className="text-xs text-slate-400 mt-1 block">
                VAD sliced by Faster-Whisper INT8
              </span>
            </div>

            {/* Sarcasm / Acoustic Incongruity */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Acoustic Friction / Sarcasm</span>
              <div className="flex items-center space-x-2">
                <ShieldAlert
                  className={`w-5 h-5 ${
                    pipelineData.sarcastic_segments_count > 0
                      ? 'text-rose-400'
                      : 'text-emerald-400'
                  }`}
                />
                <span className="text-xl sm:text-2xl font-bold text-white">
                  {pipelineData.sarcastic_segments_count}
                </span>
              </div>
              <span className="text-xs text-slate-400 mt-1 block">
                {pipelineData.sarcastic_segments_count > 0
                  ? 'Detected vocal contradiction'
                  : 'Zero incongruity detected'}
              </span>
            </div>
          </div>

          {/* Probability Distribution Chart & Segment Flow */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <EmotionBarChart
                probabilities={pipelineData.global_probability_vector}
                timeDistribution={pipelineData.time_distribution_percent}
              />
            </div>

            {/* Segment Flow & Filters */}
            <div className="lg:col-span-2 space-y-4">
              {/* Filter & Search Header */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-3 rounded-2xl bg-slate-900/90 border border-slate-800">
                {/* Search Bar */}
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    placeholder="Search transcribed text..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                  />
                </div>

                {/* Emotion Filter */}
                <div className="flex items-center space-x-2 text-xs">
                  <Filter className="w-4 h-4 text-slate-500" />
                  <select
                    value={emotionFilter}
                    onChange={(e) => setEmotionFilter(e.target.value)}
                    className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="all">All Emotions ({pipelineData.segments.length})</option>
                    {Object.entries(CANONICAL_EMOTIONS).map(([k, m]) => (
                      <option key={k} value={k}>
                        {m.label} (
                        {pipelineData.segment_counts[k as keyof typeof CANONICAL_EMOTIONS] || 0})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Segment Cards List */}
              <div className="space-y-3 max-h-[680px] overflow-y-auto pr-1">
                {filteredSegments.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 bg-slate-900/40 rounded-2xl border border-dashed border-slate-800">
                    No speech segments match the search criteria.
                  </div>
                ) : (
                  filteredSegments.map((segment) => (
                    <SegmentCard
                      key={segment.segment_index}
                      segment={segment}
                      isActive={activeSegment?.segment_index === segment.segment_index}
                      isPlaying={false}
                      onPlaySegment={handlePlaySegment}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
