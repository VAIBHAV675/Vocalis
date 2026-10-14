import React from 'react';
import { Activity, Mic, Layers, Info, Cpu, Sparkles, Award } from 'lucide-react';

export type TabType = 'inspector' | 'studio' | 'clips' | 'ser_validation' | 'architecture';

interface NavbarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  systemInfo?: {
    device?: string;
    vram_used_mb?: number;
    vram_total_mb?: number;
  };
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, systemInfo }) => {
  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/80 border-b border-slate-800/80 shadow-2xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onTabChange('inspector')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Activity className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  Vocalis
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  Multimodal SER
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono hidden sm:block">
                Whisper INT8 + Emotion2Vec+ + Kev-0.8B
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => onTabChange('inspector')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'inspector'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span className="hidden md:inline">Pipeline Inspector</span>
              <span className="md:hidden">Inspector</span>
            </button>

            <button
              onClick={() => onTabChange('studio')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'studio'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <Mic className="w-4 h-4" />
              <span className="hidden md:inline">Live Studio</span>
              <span className="md:hidden">Studio</span>
            </button>

            <button
              onClick={() => onTabChange('clips')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'clips'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span className="hidden md:inline">2,000+ Clip Explorer</span>
              <span className="md:hidden">Clips (2K+)</span>
            </button>

            <button
              onClick={() => onTabChange('ser_validation')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'ser_validation'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <Award className="w-4 h-4 text-amber-400" />
              <span className="hidden md:inline">SER Validation (122)</span>
              <span className="md:hidden">SER Demo</span>
            </button>

            <button
              onClick={() => onTabChange('architecture')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'architecture'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <Info className="w-4 h-4" />
              <span className="hidden md:inline">Architecture & Benchmarks</span>
              <span className="md:hidden">Benchmarks</span>
            </button>
          </nav>

          {/* System Hardware Status */}
          <div className="hidden lg:flex items-center space-x-3 text-xs">
            <div className="flex items-center space-x-2 px-2.5 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>{systemInfo?.device || 'GTX 1650 4GB'}</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <div className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-indigo-950/40 border border-indigo-800/40 text-indigo-300">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>INT8 ASR + NF4 LLM</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
