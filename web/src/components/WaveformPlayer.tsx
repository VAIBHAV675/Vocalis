import React, { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { Play, Pause, RotateCcw, RotateCw, Volume2, VolumeX, FastForward } from 'lucide-react';
import { formatSeconds } from '../utils/audio';

export interface WaveformPlayerHandle {
  seekTo: (timeSec: number, playImmediately?: boolean, stopAtSec?: number) => void;
  pause: () => void;
  play: () => void;
}

interface WaveformPlayerProps {
  audioUrl: string;
  onTimeUpdate?: (currentTime: number) => void;
  activeSegmentRange?: { start: number; end: number } | null;
  className?: string;
  title?: string;
}

export const WaveformPlayer = forwardRef<WaveformPlayerHandle, WaveformPlayerProps>(
  ({ audioUrl, onTimeUpdate, activeSegmentRange, className = '', title }, ref) => {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [playbackRate, setPlaybackRate] = useState(1);
    const [volume, setVolume] = useState(1);
    const [isMuted, setIsMuted] = useState(false);
    const [stopAtTime, setStopAtTime] = useState<number | null>(null);

    useImperativeHandle(ref, () => ({
      seekTo: (timeSec: number, playImmediately = true, stopAtSec?: number) => {
        if (audioRef.current) {
          audioRef.current.currentTime = timeSec;
          setCurrentTime(timeSec);
          setStopAtTime(stopAtSec ?? null);
          if (playImmediately) {
            audioRef.current.play().then(() => setIsPlaying(true)).catch(console.error);
          }
        }
      },
      pause: () => {
        if (audioRef.current) {
          audioRef.current.pause();
          setIsPlaying(false);
        }
      },
      play: () => {
        if (audioRef.current) {
          audioRef.current.play().then(() => setIsPlaying(true)).catch(console.error);
        }
      },
    }));

    useEffect(() => {
      // Reset state when audioUrl changes
      setIsPlaying(false);
      setCurrentTime(0);
      setStopAtTime(null);
    }, [audioUrl]);

    const handleTimeUpdate = () => {
      if (!audioRef.current) return;
      const t = audioRef.current.currentTime;
      setCurrentTime(t);
      if (onTimeUpdate) {
        onTimeUpdate(t);
      }

      // Check if we should stop at segment end
      if (stopAtTime !== null && t >= stopAtTime) {
        audioRef.current.pause();
        setIsPlaying(false);
        setStopAtTime(null);
      }
    };

    const handleLoadedMetadata = () => {
      if (audioRef.current) {
        setDuration(audioRef.current.duration || 0);
      }
    };

    const togglePlayPause = () => {
      if (!audioRef.current) return;
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        // If stopped at end of segment, clear segment stop so user can continue listening
        setStopAtTime(null);
        audioRef.current.play().then(() => setIsPlaying(true)).catch(console.error);
      }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
      const targetTime = parseFloat(e.target.value);
      if (audioRef.current) {
        audioRef.current.currentTime = targetTime;
        setCurrentTime(targetTime);
        setStopAtTime(null);
      }
    };

    const skipTime = (delta: number) => {
      if (!audioRef.current) return;
      const newTime = Math.max(0, Math.min(duration, audioRef.current.currentTime + delta));
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
      setStopAtTime(null);
    };

    const changePlaybackRate = (rate: number) => {
      if (audioRef.current) {
        audioRef.current.playbackRate = rate;
        setPlaybackRate(rate);
      }
    };

    const toggleMute = () => {
      if (!audioRef.current) return;
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    };

    const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = parseFloat(e.target.value);
      if (audioRef.current) {
        audioRef.current.volume = v;
        setVolume(v);
        setIsMuted(v === 0);
      }
    };

    const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
      <div className={`p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl ${className}`}>
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />

        {title && (
          <div className="flex items-center justify-between mb-3 text-sm">
            <span className="font-semibold text-slate-200 truncate">{title}</span>
            {activeSegmentRange && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono">
                Active: {formatSeconds(activeSegmentRange.start)} - {formatSeconds(activeSegmentRange.end)}
              </span>
            )}
          </div>
        )}

        {/* Progress Bar & Timeline */}
        <div className="space-y-1 mb-4">
          <div className="relative w-full h-3 bg-slate-800 rounded-full cursor-pointer overflow-hidden group">
            {/* Background Track */}
            <div
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 transition-all rounded-full"
              style={{ width: `${progressPercent}%` }}
            />
            {/* Active Segment Marker on track */}
            {activeSegmentRange && duration > 0 && (
              <div
                className="absolute top-0 h-full bg-emerald-400/40 border-l border-r border-emerald-400"
                style={{
                  left: `${(activeSegmentRange.start / duration) * 100}%`,
                  width: `${Math.max(1, ((activeSegmentRange.end - activeSegmentRange.start) / duration) * 100)}%`,
                }}
              />
            )}
            <input
              type="range"
              min={0}
              max={duration || 100}
              step={0.05}
              value={currentTime}
              onChange={handleSeek}
              className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
            />
          </div>

          <div className="flex justify-between text-xs font-mono text-slate-400">
            <span>{formatSeconds(currentTime)}</span>
            <span>{formatSeconds(duration)}</span>
          </div>
        </div>

        {/* Controls Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Main Transport */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => skipTime(-5)}
              title="Rewind 5s"
              className="p-2 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={togglePlayPause}
              className="w-11 h-11 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white flex items-center justify-center shadow-lg shadow-indigo-600/30 transition"
            >
              {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
            </button>

            <button
              onClick={() => skipTime(5)}
              title="Forward 5s"
              className="p-2 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition"
            >
              <RotateCw className="w-4 h-4" />
            </button>
          </div>

          {/* Speed Selector */}
          <div className="flex items-center space-x-1 bg-slate-950/60 p-1 rounded-xl border border-slate-800 text-xs">
            <FastForward className="w-3.5 h-3.5 text-slate-500 ml-1.5 mr-0.5" />
            {[0.75, 1, 1.25, 1.5].map((rate) => (
              <button
                key={rate}
                onClick={() => changePlaybackRate(rate)}
                className={`px-2 py-1 rounded-lg font-medium transition ${
                  playbackRate === rate
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>

          {/* Volume Control */}
          <div className="flex items-center space-x-2 text-slate-400">
            <button onClick={toggleMute} className="hover:text-slate-100 transition">
              {isMuted || volume === 0 ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-16 sm:w-20 h-1.5 bg-slate-800 accent-indigo-500 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>
    );
  }
);
