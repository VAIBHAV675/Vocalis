import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, Play, Pause, Save, CheckCircle2, Sparkles, RefreshCw, Layers } from 'lucide-react';
import { WavRecorder, formatSeconds } from '../utils/audio';
import type { PipelineResult, SavedRecordingItem } from '../types';
import { SENTIMENT_META, emotionToSentiment } from '../types';
import { SegmentCard } from '../components/SegmentCard';
import { EmotionBarChart } from '../components/EmotionBarChart';

interface LiveStudioViewProps {
  onInspectResult: (result: PipelineResult) => void;
}

export const LiveStudioView: React.FC<LiveStudioViewProps> = ({ onInspectResult }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<'en' | 'ja'>('en');
  const [decisionModel, setDecisionModel] = useState<'kev-4b' | 'kev-0.8b'>('kev-4b');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<PipelineResult | null>(null);
  const [saveTitle, setSaveTitle] = useState('');
  const [isSaved, setIsSaved] = useState(false);
  const [savedHistory, setSavedHistory] = useState<SavedRecordingItem[]>([]);

  const recorderRef = useRef<WavRecorder | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const audioPreviewRef = useRef<HTMLAudioElement | null>(null);
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);

  // Load saved recordings on mount
  useEffect(() => {
    fetchSavedHistory();
  }, []);

  const fetchSavedHistory = () => {
    fetch('/api/saved_recordings')
      .then((res) => res.json())
      .then((data) => setSavedHistory(data))
      .catch((err) => console.error('Failed to load saved recordings:', err));
  };

  // Start live microphone recording
  const startRecording = async () => {
    try {
      setRecordedBlob(null);
      setAudioUrl(null);
      setUploadedFile(null);
      setAnalysisResult(null);
      setIsSaved(false);
      setRecordDuration(0);

      const rec = new WavRecorder();
      recorderRef.current = rec;
      const analyser = await rec.start();
      setIsRecording(true);

      // Start timer
      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        setRecordDuration((Date.now() - startTime) / 1000);
      }, 100);

      // Start canvas waveform visualizer
      drawWaveform(analyser);
    } catch (err) {
      console.error('Failed to access microphone:', err);
      alert('Microphone access denied or unavailable. Please check browser permissions.');
    }
  };

  // Stop recording
  const stopRecording = async () => {
    if (!recorderRef.current) return;
    setIsRecording(false);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    const blob = await recorderRef.current.stop();
    setRecordedBlob(blob);
    const url = URL.createObjectURL(blob);
    setAudioUrl(url);
    setSaveTitle(`Microphone Recording (${new Date().toLocaleTimeString()})`);
  };

  // Draw real-time canvas frequency visualizer
  const drawWaveform = (analyser: AnalyserNode) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      animFrameRef.current = requestAnimationFrame(render);
      analyser.getByteFrequencyData(dataArray);

      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height * 0.9;

        // Gradient bar
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.5, '#a855f7');
        gradient.addColorStop(1, '#ec4899');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);

        x += barWidth + 1;
        if (x > canvas.width) break;
      }
    };
    render();
  };

  // Handle file drop/upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setRecordedBlob(null);
      setAudioUrl(URL.createObjectURL(file));
      setSaveTitle(file.name.replace(/\.[^/.]+$/, ''));
      setAnalysisResult(null);
      setIsSaved(false);
    }
  };

  // Run finalized pipeline
  const runAnalysis = async () => {
    const fileToSend = recordedBlob || uploadedFile;
    if (!fileToSend) return;

    setAnalyzing(true);
    setAnalysisStep('Uploading audio and initializing Faster-Whisper Large-v3 Turbo (INT8)...');

    try {
      const formData = new FormData();
      formData.append('file', fileToSend, uploadedFile?.name || 'microphone_recording.wav');
      formData.append('language', language);
      formData.append('model', decisionModel);

      const stepTimer1 = setTimeout(() => {
        setAnalysisStep('Transcribing speech segments & extracting Silero VAD timestamps...');
      }, 1500);

      const stepTimer2 = setTimeout(() => {
        setAnalysisStep('Chopping segments & computing Emotion2Vec+ Large (1024-d) + Librosa Prosody...');
      }, 4000);

      const stepTimer3 = setTimeout(() => {
        setAnalysisStep(
          decisionModel === 'kev-4b'
            ? 'Running Kev-4B INT4 (GGUF Q4_K_M) decision model & arbitrating multimodal state...'
            : 'Running Kev-0.8B INT4 (BitsAndBytes NF4) decision model & arbitrating multimodal state...'
        );
      }, 7000);

      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const data: PipelineResult = await res.json();
      setAnalysisResult(data);
      setAnalyzing(false);
      setAnalysisStep('');
    } catch (err: unknown) {
      console.error(err);
      const msg = err instanceof Error ? err.message : 'Analysis failed';
      alert(`Pipeline execution error: ${msg}`);
      setAnalyzing(false);
      setAnalysisStep('');
    }
  };

  // Save recording to persistent database
  const handleSaveResult = async () => {
    if (!analysisResult) return;
    try {
      const payload = {
        title: saveTitle || 'Recorded Audio',
        audio_url: analysisResult.audio_url,
        analysis: analysisResult,
      };

      const res = await fetch('/api/save_recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setIsSaved(true);
        fetchSavedHistory();
      }
    } catch (err) {
      console.error('Failed to save recording:', err);
    }
  };

  const togglePreviewPlay = () => {
    if (!audioPreviewRef.current) return;
    if (isPreviewPlaying) {
      audioPreviewRef.current.pause();
      setIsPreviewPlaying(false);
    } else {
      audioPreviewRef.current.play().then(() => setIsPreviewPlaying(true)).catch(console.error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="p-5 rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl">
        <div className="flex items-center space-x-2 mb-1">
          <span className="p-1 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Mic className="w-4 h-4" />
          </span>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
            Live Recording & Custom Audio Studio
          </h1>
        </div>
        <p className="text-xs sm:text-sm text-slate-400">
          Record live speech directly from your microphone or upload any audio file to run through the finalized Faster-Whisper INT8 + Emotion2Vec+ + Kev-0.8B INT4 pipeline.
        </p>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Capture & Control (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Microphone Card */}
          <div className="p-5 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <Mic className="w-4 h-4 text-indigo-400" />
                <span>Microphone Capture (16kHz Studio Audio)</span>
              </span>
              {isRecording && (
                <span className="flex items-center space-x-1.5 text-xs text-rose-400 font-mono animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  <span>RECORDING ({formatSeconds(recordDuration)})</span>
                </span>
              )}
            </div>

            {/* Visualizer Canvas */}
            <div className="relative w-full h-32 rounded-2xl bg-[#090d16] border border-slate-800 overflow-hidden flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={600}
                height={128}
                className="w-full h-full block"
              />
              {!isRecording && !audioUrl && (
                <div className="absolute text-center text-slate-500 text-xs">
                  Click 'Start Recording' to stream live speech into the real-time acoustic visualizer
                </div>
              )}
            </div>

            {/* Recording Transport & Action Buttons */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={analyzing}
                  className="flex items-center space-x-2 px-5 py-2.5 rounded-2xl bg-rose-600 hover:bg-rose-500 active:scale-95 text-white font-medium text-sm shadow-lg shadow-rose-600/30 transition disabled:opacity-50"
                >
                  <Mic className="w-4 h-4" />
                  <span>Start Recording</span>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex items-center space-x-2 px-5 py-2.5 rounded-2xl bg-slate-100 hover:bg-white active:scale-95 text-slate-900 font-medium text-sm shadow-lg transition"
                >
                  <Square className="w-4 h-4 fill-current text-rose-600" />
                  <span>Stop Recording ({formatSeconds(recordDuration)})</span>
                </button>
              )}

              {/* Language & Model Selectors */}
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <div className="flex items-center space-x-2">
                  <span className="text-slate-400">Target Language:</span>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as 'en' | 'ja')}
                    className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-indigo-500 font-medium"
                  >
                    <option value="en">English (Earnings22 domain)</option>
                    <option value="ja">Japanese (ReazonSpeech domain)</option>
                  </select>
                </div>

                <div className="flex items-center space-x-2">
                  <span className="text-slate-400">Decision Model:</span>
                  <select
                    value={decisionModel}
                    onChange={(e) => setDecisionModel(e.target.value as 'kev-4b' | 'kev-0.8b')}
                    className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-indigo-300 focus:outline-none focus:border-indigo-500 font-medium"
                  >
                    <option value="kev-4b">Kev-4B INT4 (Default · High-Capacity)</option>
                    <option value="kev-0.8b">Kev-0.8B INT4 (Fast Lightweight)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Audio Upload Alternative */}
          <div className="p-5 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
            <span className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <Upload className="w-4 h-4 text-indigo-400" />
              <span>Or Upload Custom Audio File (.wav, .mp3, .ogg)</span>
            </span>

            <label className="flex flex-col items-center justify-center w-full h-24 border-2 border-dashed border-slate-800 rounded-2xl cursor-pointer hover:border-indigo-500/60 hover:bg-slate-950/40 transition">
              <div className="flex flex-col items-center justify-center pt-2 pb-2 text-xs text-slate-400">
                <Upload className="w-5 h-5 text-slate-500 mb-1" />
                <span>
                  {uploadedFile ? (
                    <strong className="text-indigo-400">{uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)</strong>
                  ) : (
                    'Click to select or drag audio file here'
                  )}
                </span>
              </div>
              <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          {/* Preview & Pipeline Execution Card */}
          {audioUrl && (
            <div className="p-5 rounded-3xl bg-slate-900/90 border border-indigo-500/30 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-white">Audio Ready for Analysis</span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={togglePreviewPlay}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition"
                  >
                    {isPreviewPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current ml-0.5" />}
                    <span>{isPreviewPlaying ? 'Pause' : 'Preview Audio'}</span>
                  </button>
                  <audio
                    ref={audioPreviewRef}
                    src={audioUrl}
                    onEnded={() => setIsPreviewPlaying(false)}
                    className="hidden"
                  />
                </div>
              </div>

              {/* Run Analysis Button */}
              <button
                onClick={runAnalysis}
                disabled={analyzing}
                className="w-full flex items-center justify-center space-x-2 py-3.5 px-4 rounded-2xl bg-gradient-to-r from-indigo-600 via-violet-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 text-white font-semibold text-sm shadow-xl shadow-indigo-600/30 active:scale-95 transition disabled:opacity-50"
              >
                {analyzing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing through Multimodal Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Analyze with Finalized Pipeline (Whisper INT8 + Emotion2Vec+ + {decisionModel === 'kev-4b' ? 'Kev-4B' : 'Kev-0.8B'} INT4)</span>
                  </>
                )}
              </button>

              {/* Live Step Progress Status */}
              {analyzing && (
                <div className="p-3.5 rounded-2xl bg-indigo-950/40 border border-indigo-800/40 text-xs text-indigo-300 flex items-center space-x-3">
                  <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />
                  <span className="font-mono">{analysisStep}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Saved Recordings & Quick Load (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="p-5 rounded-3xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span>Saved Recordings History</span>
              </span>
              <span className="text-xs text-slate-400 font-mono">{savedHistory.length} items</span>
            </div>

            <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
              {savedHistory.length === 0 ? (
                <div className="text-center py-8 text-xs text-slate-500">
                  No saved recordings yet. Record or upload an audio file and save it to revisit here.
                </div>
              ) : (
                savedHistory.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 rounded-2xl bg-slate-950/80 border border-slate-800 hover:border-slate-700 transition space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-200 truncate">{item.title}</span>
                      <span className="text-[11px] text-slate-500 font-mono">{item.saved_at.split(' ')[0]}</span>
                    </div>

                    <div className="flex items-center justify-between text-xs gap-1.5">
                      <div className="flex flex-wrap items-center gap-1.5">
                        {(() => {
                          const sk = item.dominant_sentiment || emotionToSentiment(item.dominant_emotion);
                          const sm = SENTIMENT_META[sk] || SENTIMENT_META.neutral;
                          return (
                            <span className={`px-2 py-0.5 rounded-md border text-[10px] font-semibold uppercase ${sm.badgeBg}`}>
                              {sm.shortLabel}
                            </span>
                          );
                        })()}
                        <span className="px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 font-mono text-[11px]">
                          {item.dominant_emotion} ({Math.round(item.dominant_probability * 100)}%)
                        </span>
                      </div>

                      <button
                        onClick={async () => {
                          const res = await fetch(`/api/pipeline/${item.id}`);
                          if (res.ok) {
                            const data = await res.json();
                            setAnalysisResult(data);
                            onInspectResult(data);
                          }
                        }}
                        className="text-indigo-400 hover:text-indigo-300 font-medium text-[11px] underline"
                      >
                        Load in Inspector
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Output Section (Shows when analyzed) */}
      {analysisResult && (
        <div className="space-y-6 pt-4 border-t border-slate-800 animate-fadeIn">
          {/* Save Bar */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-4 rounded-2xl bg-indigo-950/30 border border-indigo-800/40">
            <div className="flex items-center space-x-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <div>
                <span className="text-sm font-semibold text-white">Pipeline Execution Complete</span>
                <p className="text-xs text-indigo-300">
                  Sentiment:{' '}
                  <strong className="uppercase">
                    {analysisResult.dominant_sentiment ||
                      emotionToSentiment(
                        analysisResult.dominant_emotion,
                        analysisResult.sarcastic_segments_count > 0
                      )}
                  </strong>{' '}
                  • Dominant Emotion: <strong>{analysisResult.dominant_emotion}</strong> (
                  {(analysisResult.dominant_probability * 100).toFixed(1)}%) across{' '}
                  {analysisResult.total_segments} speech segment(s).
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="text"
                placeholder="Recording title..."
                value={saveTitle}
                onChange={(e) => setSaveTitle(e.target.value)}
                className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none"
              />
              <button
                onClick={handleSaveResult}
                disabled={isSaved}
                className={`flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-medium transition ${
                  isSaved
                    ? 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 cursor-default'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-md'
                }`}
              >
                <Save className="w-3.5 h-3.5" />
                <span>{isSaved ? 'Saved!' : 'Save Results'}</span>
              </button>

              <button
                onClick={() => onInspectResult(analysisResult)}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium shadow-md transition"
              >
                Open in Full Inspector
              </button>
            </div>
          </div>

          {/* Success Banner when saved */}
          {isSaved && (
            <div className="p-4 rounded-2xl bg-gradient-to-r from-emerald-950/60 to-slate-900 border border-emerald-500/40 text-xs text-emerald-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-lg">
              <div className="flex items-center space-x-3">
                <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                <div>
                  <span className="font-semibold text-emerald-300">Successfully saved to database!</span>
                  <p className="text-[11px] text-slate-300 mt-0.5">
                    Your recording is stored permanently. You can access it in <strong>Saved Recordings History</strong> (panel above), in the <strong>Clips Explorer</strong> (Saved tab), or inspect its waveform and probabilities directly.
                  </p>
                </div>
              </div>
              <button
                onClick={() => onInspectResult(analysisResult)}
                className="px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium shadow transition whitespace-nowrap self-start sm:self-auto"
              >
                Inspect Waveform & Segments →
              </button>
            </div>
          )}

          {/* Side by side Distribution and Segments */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <EmotionBarChart
                probabilities={analysisResult.global_probability_vector}
                timeDistribution={analysisResult.time_distribution_percent}
              />
            </div>

            <div className="lg:col-span-2 space-y-3">
              <h3 className="text-sm font-semibold text-slate-300">
                Segmented Speech Analysis ({analysisResult.segments.length} segment{analysisResult.segments.length === 1 ? '' : 's'})
              </h3>
              <div className="space-y-3">
                {analysisResult.segments.map((seg) => (
                  <SegmentCard
                    key={seg.segment_index}
                    segment={seg}
                    isActive={false}
                    isPlaying={false}
                    onPlaySegment={() => onInspectResult(analysisResult)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
