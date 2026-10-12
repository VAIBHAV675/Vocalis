import React, { useState } from 'react';
import { Play, Pause, ChevronDown, ChevronUp, AlertCircle, Waves, Gauge, Zap } from 'lucide-react';
import { CANONICAL_EMOTIONS } from '../types';
import type { Segment } from '../types';
import { formatSeconds } from '../utils/audio';

interface SegmentCardProps {
  segment: Segment;
  isActive: boolean;
  isPlaying: boolean;
  onPlaySegment: (segment: Segment) => void;
}

export const SegmentCard: React.FC<SegmentCardProps> = ({
  segment,
  isActive,
  isPlaying,
  onPlaySegment,
}) => {
  const [expanded, setExpanded] = useState(false);

  const emotionMeta = CANONICAL_EMOTIONS[segment.predicted_emotion] || {
    id: segment.predicted_emotion,
    label: segment.predicted_emotion,
    color: '#64748b',
    badgeBg: 'bg-slate-800 text-slate-300 border-slate-700',
    textColor: 'text-slate-400',
  };

  const confidencePct = Math.round((segment.confidence <= 1 ? segment.confidence * 100 : segment.confidence));

  return (
    <div
      onClick={() => onPlaySegment(segment)}
      className={`relative rounded-2xl p-4 transition-all duration-300 cursor-pointer border ${
        isActive
          ? 'bg-slate-900 border-indigo-500 shadow-xl shadow-indigo-500/10 ring-2 ring-indigo-500/30'
          : 'bg-slate-900/60 border-slate-800/80 hover:bg-slate-900 hover:border-slate-700'
      }`}
    >
      {/* Top Header: Index, Time range, and Emotion Badge */}
      <div className="flex items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center space-x-2.5">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onPlaySegment(segment);
            }}
            className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
              isActive && isPlaying
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/40 animate-pulse'
                : 'bg-slate-800 text-slate-300 hover:bg-indigo-600 hover:text-white'
            }`}
            title="Play segment"
          >
            {isActive && isPlaying ? (
              <Pause className="w-4 h-4 fill-current" />
            ) : (
              <Play className="w-4 h-4 fill-current ml-0.5" />
            )}
          </button>

          <span className="font-mono text-xs text-slate-400">
            #{segment.segment_index.toString().padStart(2, '0')}
          </span>

          <span className="font-mono text-xs px-2 py-0.5 rounded-md bg-slate-950 border border-slate-800 text-slate-300">
            {formatSeconds(segment.start_sec)} → {formatSeconds(segment.end_sec)}{' '}
            <span className="text-slate-500">({segment.duration_sec.toFixed(1)}s)</span>
          </span>
        </div>

        {/* Emotion Badge */}
        <div className="flex items-center space-x-1.5">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-medium border flex items-center space-x-1.5 shadow-sm ${emotionMeta.badgeBg}`}
          >
            <span
              className="w-2 h-2 rounded-full inline-block"
              style={{ backgroundColor: emotionMeta.color }}
            />
            <span>{emotionMeta.label}</span>
            <span className="font-mono opacity-80 text-[11px]">{confidencePct}%</span>
          </span>
        </div>
      </div>

      {/* Transcript Text */}
      <p className="text-slate-100 text-sm sm:text-base font-normal leading-relaxed mb-3 pl-1">
        "{segment.transcript}"
      </p>

      {/* Prosody & Acoustic Indicators */}
      <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono text-slate-400">
        {segment.ser_acoustic_prior && (
          <span className="px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800/80 text-slate-300 flex items-center space-x-1">
            <span className="text-slate-500">SER Prior:</span>
            <span className="capitalize text-indigo-400 font-medium">{segment.ser_acoustic_prior}</span>
          </span>
        )}

        {segment.prosody?.pitch && (
          <span className="px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800/80 flex items-center space-x-1">
            <Waves className="w-3 h-3 text-cyan-400" />
            <span>{segment.prosody.pitch.split('(')[0].trim()}</span>
          </span>
        )}

        {segment.prosody?.tempo && (
          <span className="px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800/80 flex items-center space-x-1">
            <Gauge className="w-3 h-3 text-emerald-400" />
            <span>{segment.prosody.tempo.split('(')[0].trim()}</span>
          </span>
        )}

        {segment.prosody?.energy && (
          <span className="px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800/80 flex items-center space-x-1">
            <Zap className="w-3 h-3 text-amber-400" />
            <span>{segment.prosody.energy.split('(')[0].trim()}</span>
          </span>
        )}

        {segment.sarcasm_flag && (
          <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 font-medium flex items-center space-x-1">
            <AlertCircle className="w-3 h-3" />
            <span>Acoustic Friction / Sarcasm</span>
          </span>
        )}

        <button
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}
          className="ml-auto text-slate-400 hover:text-slate-200 text-xs flex items-center space-x-0.5 py-0.5 px-1.5 rounded hover:bg-slate-800 transition"
        >
          <span>{expanded ? 'Less' : 'Distribution'}</span>
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>

      {/* Collapsible Segment Probability Vector */}
      {expanded && segment.probabilities && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="mt-3 pt-3 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs"
        >
          {Object.entries(segment.probabilities).map(([key, prob]) => {
            const m = CANONICAL_EMOTIONS[key as keyof typeof CANONICAL_EMOTIONS];
            if (!m) return null;
            const pct = Math.round((prob <= 1 ? prob * 100 : prob));
            const isWinner = key === segment.predicted_emotion;
            return (
              <div
                key={key}
                className={`p-2 rounded-xl border ${
                  isWinner ? 'bg-slate-800/80 border-indigo-500/50' : 'bg-slate-950/60 border-slate-800/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] text-slate-400 truncate">{m.label.split('/')[0]}</span>
                  <span className="font-mono text-slate-200">{pct}%</span>
                </div>
                <div className="w-full h-1 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${pct}%`, backgroundColor: m.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
