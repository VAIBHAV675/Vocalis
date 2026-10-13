export type CanonicalEmotionKey =
  | 'neutral_inquiry'
  | 'satisfaction_gratitude'
  | 'frustration_irritation'
  | 'outrage_hostility'
  | 'disappointed_sad'
  | 'sarcasm_passive_aggression'
  | 'anxiety_confusion'
  | 'urgency_panic';

export interface EmotionMeta {
  id: CanonicalEmotionKey;
  label: string;
  color: string;
  bgLight: string;
  badgeBg: string;
  textColor: string;
  description: string;
}

export const CANONICAL_EMOTIONS: Record<CanonicalEmotionKey, EmotionMeta> = {
  neutral_inquiry: {
    id: 'neutral_inquiry',
    label: 'Neutral / Inquiry',
    color: '#64748b',
    bgLight: 'bg-slate-500/10',
    badgeBg: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
    textColor: 'text-slate-400',
    description: 'Calm routine business inquiry, objective facts, operational query.',
  },
  satisfaction_gratitude: {
    id: 'satisfaction_gratitude',
    label: 'Satisfaction / Gratitude',
    color: '#10b981',
    bgLight: 'bg-emerald-500/10',
    badgeBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    textColor: 'text-emerald-400',
    description: 'Delight, genuine satisfaction, heartfelt appreciation, relief.',
  },
  frustration_irritation: {
    id: 'frustration_irritation',
    label: 'Frustration / Irritation',
    color: '#f59e0b',
    bgLight: 'bg-amber-500/10',
    badgeBg: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    textColor: 'text-amber-400',
    description: 'Annoyance, impatient complaints, repeated friction, exasperation.',
  },
  outrage_hostility: {
    id: 'outrage_hostility',
    label: 'Outrage / Hostility',
    color: '#ef4444',
    bgLight: 'bg-rose-500/10',
    badgeBg: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    textColor: 'text-rose-400',
    description: 'Extreme fury, aggressive hostility, intense confrontation.',
  },
  disappointed_sad: {
    id: 'disappointed_sad',
    label: 'Disappointed / Sad',
    color: '#6366f1',
    bgLight: 'bg-indigo-500/10',
    badgeBg: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    textColor: 'text-indigo-400',
    description: 'Sorrow, grief, emotional dejection, heavy sighing, hopeless tone.',
  },
  sarcasm_passive_aggression: {
    id: 'sarcasm_passive_aggression',
    label: 'Sarcasm / Passive-Aggressive',
    color: '#ec4899',
    bgLight: 'bg-pink-500/10',
    badgeBg: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
    textColor: 'text-pink-400',
    description: 'Ironic praise, mocking cynicism, polite words concealing hostility.',
  },
  anxiety_confusion: {
    id: 'anxiety_confusion',
    label: 'Anxiety / Confusion',
    color: '#8b5cf6',
    bgLight: 'bg-purple-500/10',
    badgeBg: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    textColor: 'text-purple-400',
    description: 'Nervous uncertainty, trembling hesitation, worried distress.',
  },
  urgency_panic: {
    id: 'urgency_panic',
    label: 'Urgency / Panic',
    color: '#f43f5e',
    bgLight: 'bg-red-500/10',
    badgeBg: 'bg-red-500/20 text-red-300 border-red-500/30',
    textColor: 'text-red-400',
    description: 'Critical crisis, frantic alarm, immediate emergency demand.',
  },
};

export type SentimentKey = 'positive' | 'neutral' | 'negative';

export interface SentimentMeta {
  id: SentimentKey;
  label: string;
  shortLabel: string;
  color: string;
  badgeBg: string;
  textColor: string;
}

export const SENTIMENT_META: Record<SentimentKey, SentimentMeta> = {
  positive: {
    id: 'positive',
    label: 'Positive Sentiment',
    shortLabel: 'Positive',
    color: '#10b981',
    badgeBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    textColor: 'text-emerald-400',
  },
  neutral: {
    id: 'neutral',
    label: 'Neutral Sentiment',
    shortLabel: 'Neutral',
    color: '#94a3b8',
    badgeBg: 'bg-slate-500/20 text-slate-300 border-slate-500/40',
    textColor: 'text-slate-300',
  },
  negative: {
    id: 'negative',
    label: 'Negative Sentiment',
    shortLabel: 'Negative',
    color: '#f43f5e',
    badgeBg: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    textColor: 'text-rose-400',
  },
};

export function emotionToSentiment(emotion: string, sarcasmFlag?: boolean): SentimentKey {
  if (sarcasmFlag) return 'negative';
  if (emotion === 'satisfaction_gratitude') return 'positive';
  if (emotion === 'neutral_inquiry') return 'neutral';
  return 'negative';
}

export function computeSentimentProbabilities(
  probs?: Record<string, number>,
  sarcasmProb?: number,
  sarcasmFlag?: boolean
): Record<SentimentKey, number> {
  if (!probs) return { positive: 0, neutral: 1, negative: 0 };
  let pPos = Number(probs.satisfaction_gratitude || 0);
  const pNeu = Number(probs.neutral_inquiry || 0);
  const negKeys = [
    'frustration_irritation',
    'outrage_hostility',
    'disappointed_sad',
    'sarcasm_passive_aggression',
    'anxiety_confusion',
    'urgency_panic',
  ];
  let pNeg = negKeys.reduce((acc, k) => acc + Number(probs[k] || 0), 0);
  if (sarcasmFlag || (sarcasmProb && sarcasmProb > 0.45)) {
    const shift = pPos * Math.max(Number(sarcasmProb || 0.5), 0.5);
    pPos -= shift;
    pNeg += shift;
  }
  const tot = pPos + pNeu + pNeg;
  if (tot > 0) {
    return {
      positive: pPos / tot,
      neutral: pNeu / tot,
      negative: pNeg / tot,
    };
  }
  return { positive: 0, neutral: 1, negative: 0 };
}

export interface ProsodyMetrics {
  pitch?: string;
  tempo?: string;
  energy?: string;
  voice_quality?: string;
}

export interface Segment {
  segment_index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  transcript: string;
  ser_acoustic_prior: string;
  ser_confidence: number;
  prosody: ProsodyMetrics;
  predicted_emotion: CanonicalEmotionKey;
  predicted_sentiment?: SentimentKey;
  confidence: number;
  probabilities: Record<CanonicalEmotionKey, number>;
  sentiment_probabilities?: Record<SentimentKey, number>;
  sarcasm_probability?: number;
  sarcasm_flag?: boolean;
  reasoning?: string;
  ground_truth_emotion?: string;
  ground_truth_sentiment?: SentimentKey;
  kev_08b_prediction?: string;
  kev_4b_prediction?: string;
  latency_ms?: number;
}

export interface PipelineResult {
  id?: string;
  clip_id?: string;
  file: string;
  title?: string;
  audio_url?: string;
  language: 'en' | 'ja';
  total_duration_sec: number;
  spoken_duration_sec: number;
  total_segments: number;
  dominant_emotion: CanonicalEmotionKey;
  dominant_probability: number;
  dominant_sentiment?: SentimentKey;
  dominant_sentiment_probability?: number;
  ground_truth_emotion?: string;
  ground_truth_sentiment?: SentimentKey;
  kev_08b_prediction?: string;
  kev_4b_prediction?: string;
  latency_ms?: number;
  time_distribution_percent: Record<CanonicalEmotionKey, number>;
  sentiment_time_distribution_percent?: Record<SentimentKey, number>;
  segment_counts: Record<CanonicalEmotionKey, number>;
  sentiment_counts?: Record<SentimentKey, number>;
  global_probability_vector: Record<CanonicalEmotionKey, number>;
  global_sentiment_probability_vector?: Record<SentimentKey, number>;
  sarcastic_segments_count: number;
  segments: Segment[];
}

export interface ClipItem {
  id: string;
  title: string;
  filename: string;
  audio_url: string;
  language: 'en' | 'ja';
  type: 'single' | 'combined' | 'saved';
  duration_sec: number;
  domain: string;
  source_id?: string;
  speaker_info?: string;
  ground_truth_emotion: string;
  ground_truth_sentiment?: SentimentKey;
  ground_truth_transcript: string;
  predicted_emotion: string;
  predicted_sentiment?: SentimentKey;
  confidence: number;
  kev_08b_prediction?: string;
  kev_4b_prediction?: string;
  kev_08b_match?: boolean;
  kev_4b_match?: boolean;
  kev_08b_sentiment_match?: boolean;
  kev_4b_sentiment_match?: boolean;
  kev_08b_latency_ms?: number;
  kev_4b_latency_ms?: number;
  ser_prior?: string;
  ser_confidence?: number;
  sarcasm_probability?: number;
  latency_ms?: number;
  prosody_summary?: ProsodyMetrics | null;
  match: boolean;
  sentiment_match?: boolean;
  is_continuous: boolean;
  is_audited_60?: boolean;
  has_pipeline_results: boolean;
  time_share_percent?: number;
  saved_at?: string;
}

export interface SavedRecordingItem {
  id: string;
  title: string;
  saved_at: string;
  duration_sec: number;
  dominant_emotion: string;
  dominant_sentiment?: SentimentKey;
  dominant_probability: number;
  language: string;
  audio_url: string;
}

export interface BenchmarkSummary {
  english_accuracy: number;
  japanese_accuracy: number;
  overall_accuracy: number;
  english_sentiment_accuracy?: number;
  japanese_sentiment_accuracy?: number;
  overall_sentiment_accuracy?: number;
  english_60_accuracy?: number;
  japanese_60_accuracy?: number;
  overall_60_accuracy?: number;
  english_60_sentiment_accuracy?: number;
  japanese_60_sentiment_accuracy?: number;
  overall_60_sentiment_accuracy?: number;
  kev_08b_2000_en_acc?: number;
  kev_08b_2000_ja_acc?: number;
  kev_08b_2000_overall_acc?: number;
  kev_08b_2000_overall_sentiment_acc?: number;
  kev_4b_2000_en_acc?: number;
  kev_4b_2000_ja_acc?: number;
  kev_4b_2000_overall_acc?: number;
  kev_4b_2000_overall_sentiment_acc?: number;
  ser_2000_en_acc?: number;
  ser_2000_ja_acc?: number;
  ser_2000_overall_acc?: number;
  ser_2000_overall_sentiment_acc?: number;
  cpp_cuda_kev08_ms?: number;
  cpp_cuda_kev4b_ms?: number;
  whisper_turbo_int8_en_ms?: number;
  whisper_turbo_int8_ja_ms?: number;
  whisper_turbo_rtf_en?: number;
  whisper_turbo_rtf_ja?: number;
  asr_model: string;
  ser_model: string;
  decision_model: string;
}
