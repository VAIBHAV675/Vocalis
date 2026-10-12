// Web Audio API recorder producing 16kHz Mono 16-bit PCM WAV Blob

export class WavRecorder {
  private audioCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private analyser: AnalyserNode | null = null;
  private audioData: Float32Array[] = [];
  private recordingLength = 0;
  private isRecording = false;

  public async start(): Promise<AnalyserNode> {
    this.audioData = [];
    this.recordingLength = 0;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Create AudioContext at 16000Hz if possible, else resample
    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.audioCtx = new AudioContextClass({ sampleRate: 16000 });

    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 256;

    this.source = this.audioCtx.createMediaStreamSource(this.stream);
    // Buffer size 4096, 1 input channel, 1 output channel
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      const clone = new Float32Array(input.length);
      clone.set(input);
      this.audioData.push(clone);
      this.recordingLength += input.length;
    };

    this.source.connect(this.analyser);
    this.source.connect(this.processor);
    this.processor.connect(this.audioCtx.destination);
    this.isRecording = true;

    return this.analyser;
  }

  public stop(): Promise<Blob> {
    return new Promise((resolve) => {
      this.isRecording = false;

      if (this.source) this.source.disconnect();
      if (this.processor) this.processor.disconnect();
      if (this.stream) {
        this.stream.getTracks().forEach((t) => t.stop());
      }

      const sampleRate = this.audioCtx ? this.audioCtx.sampleRate : 16000;
      if (this.audioCtx) {
        this.audioCtx.close();
      }

      // Merge buffers
      const merged = new Float32Array(this.recordingLength);
      let offset = 0;
      for (const chunk of this.audioData) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      // Encode PCM WAV
      const wavBlob = encodeWAV(merged, sampleRate);
      resolve(wavBlob);
    });
  }

  public getAnalyser(): AnalyserNode | null {
    return this.analyser;
  }
}

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* RIFF chunk length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count: 1 */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample: 16 */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  // Write PCM samples
  let index = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(index, s, true);
    index += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

export function formatSeconds(sec: number): string {
  if (isNaN(sec) || sec < 0) return '00:00.0';
  const mins = Math.floor(sec / 60);
  const remainingSec = sec % 60;
  const secs = Math.floor(remainingSec);
  const tenths = Math.floor((remainingSec - secs) * 10);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${tenths}`;
}

export function formatTimeShort(sec: number): string {
  if (isNaN(sec) || sec < 0) return '0:00';
  const mins = Math.floor(sec / 60);
  const secs = Math.floor(sec % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
