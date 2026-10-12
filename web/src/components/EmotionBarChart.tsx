import React, { useState } from 'react';
import { CANONICAL_EMOTIONS } from '../types';
import type { CanonicalEmotionKey } from '../types';
import { BarChart3, Clock } from 'lucide-react';

interface EmotionBarChartProps {
  probabilities: Record<string, number>;
  timeDistribution?: Record<string, number>;
  compact?: boolean;
}

export const EmotionBarChart: React.FC<EmotionBarChartProps> = ({
  probabilities,
  timeDistribution,
  compact = false,
}) => {
  const [viewMode, setViewMode] = useState<'probability' | 'time'>('probability');

  const dataSource = viewMode === 'probability' || !timeDistribution ? probabilities : timeDistribution;

  // Sort emotions by value descending
  const sortedEmotions = (Object.keys(CANONICAL_EMOTIONS) as CanonicalEmotionKey[]).map((key) => {
    const val = dataSource[key] ?? 0;
    // If probability is 0-1, convert to percentage for display
    const percent = viewMode === 'probability' ? (val <= 1 ? val * 100 : val) : val;
    return {
      key,
      meta: CANONICAL_EMOTIONS[key],
      value: percent,
    };
  }).sort((a, b) => b.value - a.value);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            {viewMode === 'probability' ? 'Acoustic & Semantic Probability Vector' : 'Temporal Duration Share'}
          </h3>
        </div>

        {timeDistribution && !compact && (
          <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setViewMode('probability')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-md transition ${
                viewMode === 'probability'
                  ? 'bg-indigo-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart3 className="w-3 h-3" />
              <span>Probability</span>
            </button>
            <button
              onClick={() => setViewMode('time')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-md transition ${
                viewMode === 'time'
                  ? 'bg-indigo-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Clock className="w-3 h-3" />
              <span>Time-Share</span>
            </button>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {sortedEmotions.map(({ key, meta, value }, idx) => {
          const isDominant = idx === 0 && value > 0;
          return (
            <div key={key} className="group">
              <div className="flex items-center justify-between text-xs mb-1">
                <div className="flex items-center space-x-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full shadow-sm"
                    style={{ backgroundColor: meta.color }}
                  />
                  <span className={`font-medium ${isDominant ? 'text-white font-semibold' : 'text-slate-300'}`}>
                    {meta.label}
                  </span>
                  {isDominant && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      Dominant
                    </span>
                  )}
                </div>
                <span className="font-mono text-slate-400 group-hover:text-slate-200 transition">
                  {value.toFixed(1)}%
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800/80">
                <div
                  className="h-full rounded-full transition-all duration-500 ease-out"
                  style={{
                    width: `${Math.min(100, Math.max(value, 0.5))}%`,
                    backgroundColor: meta.color,
                    boxShadow: value > 10 ? `0 0 10px ${meta.color}40` : undefined,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
