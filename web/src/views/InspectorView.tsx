import React, { useState, useEffect, useRef } from 'react';
import { WaveformPlayer } from '../components/WaveformPlayer';
import type { WaveformPlayerHandle } from '../components/WaveformPlayer';
import { EmotionBarChart } from '../components/EmotionBarChart';
import { SegmentCard } from '../components/SegmentCard';
import type { PipelineResult, Segment, SavedRecordingItem } from '../types';
import { CANONICAL_EMOTIONS } from '../types';
import { Sparkles, Filter, Search, Globe, ShieldAlert, Layers, FolderHeart } from 'lucide-react';

interface InspectorViewProps {
  customResult?: PipelineResult | null;
  onSelectClipToInspect?: (clipId: string) => void;
}

export const InspectorView: React.FC<InspectorViewProps> = ({ customResult }) => {
  const [selectedStream, setSelectedStream] = useState<'en' | 'ja' | 'custom'>('en');
  const [loading, setLoading] = useState(false);
  const [pipelineData, setPipelineData] = useState<PipelineResult | null>(null);
  const [activeSegment, setActiveSegment] = useState<Segment | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [emotionFilter, setEmotionFilter] = useState<string>('all');
  const [savedRecordings, setSavedRecordings] = useState<SavedRecordingItem[]>([]);

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

  // When customResult changes, switch to custom view immediately
  useEffect(() => {
    if (customResult) {
      setSelectedStream('custom');
      setPipelineData(customResult);
    }
  }, [customResult]);

  // Load pipeline data when stream changes
  useEffect(() => {
    if (selectedStream === 'custom') {
      if (customResult) {
        setPipelineData(customResult);
      }
      return;
    }

    const clipId = selectedStream === 'ja' ? 'combined_japanese_30_broadcasts' : 'combined_english_30_calls';
    setLoading(true);

    fetch(`/api/pipeline/${clipId}`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load pipeline results');
        return res.json();
      })
      .then((data: PipelineResult) => {
        setPipelineData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [selectedStream]);

  // Load a specific saved recording into Inspector
  const handleSelectSavedRecording = async (savedId: string) => {
    if (!savedId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/pipeline/${savedId}`);
      if (res.ok) {
        const data: PipelineResult = await res.json();
        setSelectedStream('custom');
        setPipelineData(data);
      }
    } catch (err) {
      console.error('Failed to load saved recording for inspection:', err);
    } finally {
      setLoading(false);
    }
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

  const dominantMeta = pipelineData ? CANONICAL_EMOTIONS[pipelineData.dominant_emotion] : null;

  return (
    <div className="space-y-6">
      {/* Stream Selector & Overview Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl">
        <div>
          <div className="flex items-center space-x-2 mb-1">
            <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
              <Sparkles className="w-4 h-4" />
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Pipeline Inspector
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-slate-400">
            Click any segment card below to instantly seek the audio waveform and hear that specific speech excerpt.
          </p>
        </div>

        {/* Stream Pills */}
        <div className="flex items-center space-x-2 bg-slate-950/90 p-1.5 rounded-2xl border border-slate-800/80">
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

          {customResult && (
            <button
              onClick={() => setSelectedStream('custom')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-xl text-xs sm:text-sm font-medium transition ${
                selectedStream === 'custom' && (!pipelineData?.id || !pipelineData.id.startsWith('rec_'))
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span>Current Session Analysis</span>
            </button>
          )}

          {savedRecordings.length > 0 && (
            <div className="flex items-center space-x-1.5 pl-2 border-l border-slate-800">
              <FolderHeart className="w-3.5 h-3.5 text-pink-400" />
              <select
                value={selectedStream === 'custom' && pipelineData?.id ? pipelineData.id : ''}
                onChange={(e) => handleSelectSavedRecording(e.target.value)}
                className="bg-slate-900 border border-slate-700/80 text-xs sm:text-sm text-slate-200 px-2.5 py-1.5 rounded-xl focus:outline-none focus:border-indigo-500 font-medium"
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

      {!loading && pipelineData && (
        <>
          {/* Audio Waveform Player */}
          <WaveformPlayer
            ref={playerRef}
            audioUrl={pipelineData.audio_url || ''}
            title={pipelineData.title || `Continuous Multimodal Stream (${pipelineData.language.toUpperCase()})`}
            onTimeUpdate={handleTimeUpdate}
            activeSegmentRange={
              activeSegment ? { start: activeSegment.start_sec, end: activeSegment.end_sec } : null
            }
          />

          {/* Top Key Metrics Banner */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
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
                {Math.round((pipelineData.dominant_probability <= 1 ? pipelineData.dominant_probability * 100 : pipelineData.dominant_probability))}% probability
              </span>
            </div>

            {/* Time Share */}
            <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 block mb-1">Dominant Time-Share</span>
              <div className="flex items-baseline space-x-1.5">
                <span className="text-xl sm:text-2xl font-bold text-white">
                  {(pipelineData.time_distribution_percent[pipelineData.dominant_emotion] || 0).toFixed(1)}%
                </span>
                <span className="text-xs text-slate-500">of spoken audio</span>
              </div>
              <span className="text-xs text-slate-400 font-mono mt-1 block">
                {pipelineData.spoken_duration_sec.toFixed(1)}s / {pipelineData.total_duration_sec.toFixed(1)}s
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
                <ShieldAlert className={`w-5 h-5 ${pipelineData.sarcastic_segments_count > 0 ? 'text-rose-400' : 'text-emerald-400'}`} />
                <span className="text-xl sm:text-2xl font-bold text-white">
                  {pipelineData.sarcastic_segments_count}
                </span>
              </div>
              <span className="text-xs text-slate-400 mt-1 block">
                {pipelineData.sarcastic_segments_count > 0 ? 'Detected vocal contradiction' : 'Zero incongruity detected'}
              </span>
            </div>
          </div>

          {/* Probability Distribution Chart */}
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
                        {m.label} ({pipelineData.segment_counts[k as keyof typeof CANONICAL_EMOTIONS] || 0})
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
