/**
 * AudioWorklet de reproducción: cola de bloques que genera el worker de síntesis, volumen, auto-ganancia y
 * soft-clip. Pide bloques al worker para mantener `targetBlocks` en cola (latencia ≈ targetBlocks·hop).
 */
import type { FromWorklet, ToWorklet, WorkerToWorklet, WorkletToWorker } from "./protocol";

declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor): void;

const AGC_TARGET_DB = -20; // mismo nivel al que normaliza app.py
const AGC_MAX_DB = 60;
const AGC_MIN_DB = -40;
const STATS_INTERVAL_SEC = 0.05;

const dbToLin = (db: number) => Math.pow(10, db / 20);

class PlayerProcessor extends AudioWorkletProcessor {
  private worker: MessagePort | null = null;
  private queue: Float32Array[] = [];
  private readPos = 0;
  private inflight = 0;
  private targetBlocks = 3;
  private playing = false;
  private underruns = 0;
  private primed = false; // no contar como cortes los bloques que faltan al arrancar

  private volume = dbToLin(-12);
  private autoGain = true;
  private gain = 0; // ganancia aplicada (suavizada): arranca en 0 para hacer fade-in
  private agcDb = 0;
  private meanSq = 0;
  private peak = 0;
  private statsCountdown = 0;

  // Coeficientes de suavizado por muestra
  private readonly gainCoef = 1 - Math.exp(-1 / (0.02 * sampleRate));
  private readonly rmsCoef = 1 - Math.exp(-1 / (0.3 * sampleRate));

  constructor() {
    super();
    this.port.onmessage = (ev: MessageEvent<ToWorklet>) => {
      const msg = ev.data;
      if (msg.type === "connect") {
        this.worker = msg.port;
        this.worker.onmessage = (e: MessageEvent<WorkerToWorklet>) => {
          if (e.data.type !== "block") return;
          if (e.data.solicited) this.inflight = Math.max(0, this.inflight - 1);
          this.queue.push(e.data.data);
        };
      } else if (msg.type === "config") {
        if (msg.playing !== undefined && msg.playing !== this.playing) {
          this.playing = msg.playing;
          if (this.playing) {
            this.queue = [];
            this.readPos = 0;
            this.underruns = 0;
            this.primed = false;
          }
        }
        if (msg.volumeDb !== undefined) this.volume = msg.volumeDb <= -60 ? 0 : dbToLin(msg.volumeDb);
        if (msg.autoGain !== undefined) this.autoGain = msg.autoGain;
        if (msg.targetBlocks !== undefined) this.targetBlocks = Math.max(1, Math.round(msg.targetBlocks));
      }
    };
  }

  private request() {
    if (!this.worker || !this.playing) return;
    const deficit = this.targetBlocks - this.queue.length - this.inflight;
    if (deficit > 0) {
      this.inflight += deficit;
      const msg: WorkletToWorker = { type: "need", n: deficit };
      this.worker.postMessage(msg);
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    const left = out[0];
    const frames = left.length;
    const targetGain = this.playing ? this.volume * (this.autoGain ? dbToLin(this.agcDb) : 1) : 0;

    for (let i = 0; i < frames; i++) {
      let x = 0;
      if (this.playing) {
        const block = this.queue[0];
        if (block) {
          this.primed = true;
          x = block[this.readPos++];
          if (this.readPos >= block.length) {
            this.queue.shift();
            this.readPos = 0;
          }
        } else if (i === 0 && this.primed) {
          this.underruns++;
        }
      }
      if (!Number.isFinite(x)) x = 0;
      this.meanSq += this.rmsCoef * (x * x - this.meanSq);
      this.gain += this.gainCoef * (targetGain - this.gain);
      const y = Math.tanh(x * this.gain);
      left[i] = y;
      const a = Math.abs(y);
      if (a > this.peak) this.peak = a;
    }
    for (let c = 1; c < out.length; c++) out[c].set(left);

    // Auto-ganancia: lleva el RMS de la síntesis a AGC_TARGET_DB (el nivel cambia ~60 dB entre modelos y α)
    const rmsDb = 10 * Math.log10(this.meanSq + 1e-20);
    if (this.playing && this.meanSq > 1e-14) {
      const want = Math.min(AGC_MAX_DB, Math.max(AGC_MIN_DB, AGC_TARGET_DB - rmsDb));
      this.agcDb += 0.02 * (want - this.agcDb);
    }

    this.request();

    this.statsCountdown -= frames;
    if (this.statsCountdown <= 0) {
      this.statsCountdown = STATS_INTERVAL_SEC * sampleRate;
      const msg: FromWorklet = {
        type: "stats",
        rmsDb,
        peak: this.peak,
        queued: this.queue.length,
        underruns: this.underruns,
        agcDb: this.agcDb,
      };
      this.port.postMessage(msg);
      this.peak = 0;
    }
    return true;
  }
}

registerProcessor("asmrs-player", PlayerProcessor);
