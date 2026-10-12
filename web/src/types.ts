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
  confidence: number;
  probabilities: Record<CanonicalEmotionKey, number>;
  sarcasm_probability?: number;
  sarcasm_flag?: boolean;
  reasoning?: string;
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
  time_distribution_percent: Record<CanonicalEmotionKey, number>;
  segment_counts: Record<CanonicalEmotionKey, number>;
  global_probability_vector: Record<CanonicalEmotionKey, number>;
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
  ground_truth_transcript: string;
  predicted_emotion: string;
  confidence: number;
  kev_08b_prediction?: string;
  kev_4b_prediction?: string;
  ser_prior?: string;
  ser_confidence?: number;
  sarcasm_probability?: number;
  latency_ms?: number;
  match: boolean;
  is_continuous: boolean;
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
  dominant_probability: number;
  language: string;
  audio_url: string;
}

export interface BenchmarkSummary {
  english_accuracy: number;
  japanese_accuracy: number;
  overall_accuracy: number;
  asr_model: string;
  ser_model: string;
  decision_model: string;
}
