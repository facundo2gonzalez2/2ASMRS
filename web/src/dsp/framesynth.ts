import { FFT } from "./fft";
import { RtPghi, randomPhase } from "./phase";
import { RtisiLa } from "./rtisi";

/**
 * - rtisi: RTISI-LA (Griffin-Lim en tiempo real con lookahead) inicializado con RTPGHI. Mejor calidad, agrega
 *   `lookahead · hop` de latencia.
 * - pghi: RTPGHI solo (sin latencia extra).
 * - random: fase aleatoria, como el sintetizador C++.
 */
export type PhaseMode = "rtisi" | "pghi" | "random";

export interface RtisiConfig {
  lookahead: number;
  iterations: number;
}

const LN10_OVER_20 = Math.LN10 / 20;

/**
 * Síntesis por frames: salida del decoder → magnitud → fase → iFFT → ventana Hann → overlap-add.
 *
 * Cada llamada a `process` consume un frame y devuelve `hopLength` muestras nuevas. La magnitud se reconstruye
 * como en audio_utils.spectrogram2audio: |X| = 10^((clip(y,0,1)·xMax + dbMinNorm)/20), y la normalización del
 * overlap-add (dividir por Σw²/hop) es la de torch.istft, así el nivel coincide con el del audio de entrenamiento.
 */
export class FrameSynth {
  readonly winLength: number;
  readonly hopLength: number;
  readonly bins: number;
  private readonly fft: FFT;
  private readonly window: Float64Array;
  private readonly olaScale: number;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly ola: Float64Array;
  private readonly out: Float32Array;
  private readonly logMag: Float64Array;
  private readonly randPhase: Float64Array;
  private readonly pghi: RtPghi;
  private rtisi: RtisiLa | null = null;
  private lastMode: PhaseMode | null = null;

  constructor(winLength: number, hopLength: number) {
    this.winLength = winLength;
    this.hopLength = hopLength;
    this.bins = winLength / 2 + 1;
    this.fft = new FFT(winLength);
    // Hann periódica, igual a torch.hann_window
    this.window = new Float64Array(winLength);
    let sumSq = 0;
    for (let k = 0; k < winLength; k++) {
      this.window[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / winLength);
      sumSq += this.window[k] ** 2;
    }
    this.olaScale = hopLength / sumSq;
    this.re = new Float64Array(winLength);
    this.im = new Float64Array(winLength);
    this.ola = new Float64Array(winLength);
    this.out = new Float32Array(hopLength);
    this.logMag = new Float64Array(this.bins);
    this.randPhase = new Float64Array(this.bins);
    this.pghi = new RtPghi(winLength, hopLength);
  }

  reset(): void {
    this.ola.fill(0);
    this.pghi.reset();
    this.rtisi?.reset();
  }

  /** Frame del decoder (normalizado) → `hopLength` muestras. */
  process(y: Float32Array, xMax: number, dbMinNorm: number, mode: PhaseMode, rtisi?: RtisiConfig): Float32Array {
    for (let i = 0; i < this.bins; i++) {
      let v = y[i];
      v = v > 0 ? (v < 1 ? v : 1) : 0; // NaN → 0
      this.logMag[i] = (v * xMax + dbMinNorm) * LN10_OVER_20;
    }
    return this.processLogMagnitude(this.logMag, mode, rtisi);
  }

  /** Log natural de la magnitud → `hopLength` muestras (vista interna, válida hasta la próxima llamada). */
  processLogMagnitude(
    logMag: Float64Array | Float32Array,
    mode: PhaseMode,
    rtisi: RtisiConfig = { lookahead: 2, iterations: 8 },
  ): Float32Array {
    if (mode !== this.lastMode) {
      // Cada modo tiene su propio estado de overlap-add: empezar limpio
      this.reset();
      this.lastMode = mode;
    }
    if (mode === "rtisi") {
      if (!this.rtisi || this.rtisi.lookahead !== rtisi.lookahead) {
        this.rtisi = new RtisiLa(this.winLength, this.hopLength, this.window, this.olaScale, rtisi.lookahead);
      }
      this.rtisi.iterations = rtisi.iterations;
      return this.rtisi.process(logMag);
    }
    const n = this.winLength;
    const m = this.bins;
    const phase = mode === "pghi" ? this.pghi.next(logMag) : randomPhase(this.randPhase);
    const re = this.re;
    const im = this.im;
    for (let i = 0; i < m; i++) {
      // (−1)^i: pasar de fase referida al centro de la ventana a la del inicio del frame
      const mag = Math.exp(logMag[i]) * (i & 1 ? -1 : 1);
      re[i] = mag * Math.cos(phase[i]);
      im[i] = mag * Math.sin(phase[i]);
    }
    im[0] = 0;
    im[m - 1] = 0;
    for (let i = m; i < n; i++) {
      re[i] = re[n - i];
      im[i] = -im[n - i];
    }
    this.fft.transform(re, im, true);

    const ola = this.ola;
    for (let k = 0; k < n; k++) ola[k] += re[k] * this.window[k] * this.olaScale;
    const hop = this.hopLength;
    for (let k = 0; k < hop; k++) this.out[k] = ola[k];
    ola.copyWithin(0, hop);
    ola.fill(0, n - hop);
    return this.out;
  }
}
