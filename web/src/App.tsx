import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import type { TabType } from './components/Navbar';
import { InspectorView } from './views/InspectorView';
import { LiveStudioView } from './views/LiveStudioView';
import { ClipsView } from './views/ClipsView';
import { ArchitectureView } from './views/ArchitectureView';
import type { PipelineResult } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>(() => {
    try {
      const saved = localStorage.getItem('vocalis_active_tab');
      if (saved && ['inspector', 'studio', 'clips', 'architecture'].includes(saved)) {
        return saved as TabType;
      }
    } catch {
      // ignore
    }
    return 'inspector';
  });

  const [customResult, setCustomResult] = useState<PipelineResult | null>(() => {
    try {
      const saved = sessionStorage.getItem('vocalis_custom_result');
      if (saved) return JSON.parse(saved);
    } catch {
      // ignore
    }
    return null;
  });

  const [systemInfo, setSystemInfo] = useState<{
    device?: string;
    vram_used_mb?: number;
    vram_total_mb?: number;
  }>({});

  useEffect(() => {
    fetch('/api/system_info')
      .then((res) => res.json())
      .then((data) => {
        setSystemInfo({
          device: data.device,
          vram_used_mb: data.vram_used_mb,
          vram_total_mb: data.vram_total_mb,
        });
      })
      .catch((err) => console.error('Failed to load system info:', err));
  }, []);

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
    try {
      localStorage.setItem('vocalis_active_tab', tab);
    } catch {
      // ignore
    }
  };

  const handleInspectClip = async (clipId: string) => {
    try {
      const res = await fetch(`/api/pipeline/${clipId}`);
      if (res.ok) {
        const data: PipelineResult = await res.json();
        setCustomResult(data);
        try {
          sessionStorage.setItem('vocalis_custom_result', JSON.stringify(data));
        } catch {
          // ignore
        }
        handleTabChange('inspector');
      }
    } catch (err) {
      console.error('Failed to load clip for inspection:', err);
    }
  };

  const handleInspectResult = (result: PipelineResult) => {
    setCustomResult(result);
    try {
      sessionStorage.setItem('vocalis_custom_result', JSON.stringify(result));
    } catch {
      // ignore
    }
    handleTabChange('inspector');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
      <Navbar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        systemInfo={systemInfo}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'inspector' && (
          <InspectorView customResult={customResult} onSelectClipToInspect={handleInspectClip} />
        )}

        {activeTab === 'studio' && (
          <LiveStudioView onInspectResult={handleInspectResult} />
        )}

        {activeTab === 'clips' && (
          <ClipsView onInspectClip={handleInspectClip} />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureView />
        )}
      </main>

      <footer className="mt-auto border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Vocalis Multimodal SER Engine Active</span>
            <span className="text-slate-600">•</span>
            <span className="font-mono">Faster-Whisper Large-v3 Turbo (INT8)</span>
          </div>
          <div className="text-slate-500 font-mono">
            Emotion2Vec+ Large • Librosa DSP • Kev-0.8B INT4
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
