import { FFT } from "./fft";
import { RtPghi } from "./phase";

/**
 * RTISI-LA: Real-Time Iterative Spectrogram Inversion with Look-Ahead (Zhu, Beauregard & Wyse, 2007).
 *
 * Griffin-Lim en tiempo real: se refinan las fases de los últimos `lookahead + 1` frames contra la señal ya
 * comprometida, y recién después se emite el frame más viejo. La fase inicial de cada frame nuevo sale de RTPGHI
 * (Průša & Holighaus, 2017), que acelera mucho la convergencia. Agrega `lookahead · hop` muestras de latencia.
 */
export class RtisiLa {
  readonly lookahead: number;
  iterations: number;
  private readonly n: number;
  private readonly hop: number;
  private readonly bins: number;
  private readonly fft: FFT;
  private readonly window: Float64Array;
  private readonly olaScale: number;
  private readonly pghi: RtPghi;
  // Frames sin comprometer (el 0 es el más viejo): magnitud objetivo y fase actual como fasor unitario
  // (convención del FFT del frame). Guardar fasores evita atan2/cos/sin por bin en cada iteración.
  private readonly mags: Float64Array[];
  private readonly unitRe: Float64Array[];
  private readonly unitIm: Float64Array[];
  private readonly frames: Float64Array[]; // contribución en el tiempo de cada frame (w · ifft · escala)
  private count = 0;
  // Suma de los frames ya comprometidos, alineada al inicio del frame 0 sin comprometer
  private readonly committed: Float64Array;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly out: Float32Array;
  private readonly fiPhase: Float64Array;

  constructor(winLength: number, hopLength: number, window: Float64Array, olaScale: number, lookahead = 2, iterations = 8) {
    this.n = winLength;
    this.hop = hopLength;
    this.bins = winLength / 2 + 1;
    this.lookahead = lookahead;
    this.iterations = iterations;
    this.fft = new FFT(winLength);
    this.window = window;
    this.olaScale = olaScale;
    this.pghi = new RtPghi(winLength, hopLength);
    const slots = lookahead + 1;
    this.mags = Array.from({ length: slots }, () => new Float64Array(this.bins));
    this.unitRe = Array.from({ length: slots }, () => new Float64Array(this.bins));
    this.unitIm = Array.from({ length: slots }, () => new Float64Array(this.bins));
    this.frames = Array.from({ length: slots }, () => new Float64Array(winLength));
    this.committed = new Float64Array(winLength + lookahead * hopLength);
    this.re = new Float64Array(winLength);
    this.im = new Float64Array(winLength);
    this.out = new Float32Array(hopLength);
    this.fiPhase = new Float64Array(this.bins);
  }

  reset(): void {
    this.pghi.reset();
    this.committed.fill(0);
    this.count = 0;
  }

  /** Log-magnitud del frame nuevo → `hop` muestras (del frame de hace `lookahead` frames). */
  process(logMag: Float64Array | Float32Array): Float32Array {
    const slots = this.lookahead + 1;
    const n = this.n;
    const hop = this.hop;

    // Entra el frame nuevo en el último slot, con fase inicial de RTPGHI integrada desde la fase ya refinada del
    // frame anterior. RTPGHI usa fase referida al centro de la ventana: (−1)^m ⇔ ±π en los bins impares.
    const last = Math.min(this.count, slots - 1);
    if (last > 0) {
      const pr = this.unitRe[last - 1];
      const pi = this.unitIm[last - 1];
      for (let i = 0; i < this.bins; i++) this.fiPhase[i] = i & 1 ? Math.atan2(-pi[i], -pr[i]) : Math.atan2(pi[i], pr[i]);
      this.pghi.overrideLastPhase(this.fiPhase);
    }
    const init = this.pghi.next(logMag);
    const ur = this.unitRe[last];
    const ui = this.unitIm[last];
    for (let i = 0; i < this.bins; i++) {
      this.mags[last][i] = Math.exp(logMag[i]);
      const sign = i & 1 ? -1 : 1;
      ur[i] = sign * Math.cos(init[i]);
      ui[i] = sign * Math.sin(init[i]);
    }
    this.synthesizeFrame(last);
    if (this.count < slots) this.count++;
    const active = this.count;

    // Iteraciones de Griffin-Lim sobre los frames sin comprometer
    for (let it = 0; it < this.iterations; it++) {
      for (let j = 0; j < active; j++) {
        // Señal estimada en el segmento del frame j: comprometida + todos los frames activos que lo solapan
        const re = this.re;
        const im = this.im;
        const offset = j * hop;
        for (let k = 0; k < n; k++) re[k] = this.committed[offset + k];
        for (let q = 0; q < active; q++) {
          const shift = (q - j) * hop;
          if (Math.abs(shift) >= n) continue;
          const f = this.frames[q];
          const k0 = Math.max(0, shift);
          const k1 = Math.min(n, n + shift);
          for (let k = k0; k < k1; k++) re[k] += f[k - shift];
        }
        for (let k = 0; k < n; k++) {
          re[k] *= this.window[k];
          im[k] = 0;
        }
        this.fft.transform(re, im);
        const ur = this.unitRe[j];
        const ui = this.unitIm[j];
        for (let i = 0; i < this.bins; i++) {
          const r = Math.hypot(re[i], im[i]);
          if (r > 1e-30) {
            ur[i] = re[i] / r;
            ui[i] = im[i] / r;
          }
        }
        this.synthesizeFrame(j);
      }
    }

    if (active < slots) {
      // Todavía llenando el lookahead: silencio
      this.out.fill(0);
      return this.out;
    }

    // Comprometer el frame 0, emitir `hop` muestras y correr todo un frame
    const c = this.committed;
    const f0 = this.frames[0];
    for (let k = 0; k < n; k++) c[k] += f0[k];
    for (let k = 0; k < hop; k++) this.out[k] = c[k];
    c.copyWithin(0, hop);
    c.fill(0, c.length - hop);
    this.rotate();
    this.count = slots - 1;
    return this.out;
  }

  /** frames[j] = w · ifft(|X_j| e^{iφ_j}) · escala de overlap-add. */
  private synthesizeFrame(j: number) {
    const re = this.re;
    const im = this.im;
    const n = this.n;
    const mag = this.mags[j];
    const ur = this.unitRe[j];
    const ui = this.unitIm[j];
    for (let i = 0; i < this.bins; i++) {
      re[i] = mag[i] * ur[i];
      im[i] = mag[i] * ui[i];
    }
    im[0] = 0;
    im[this.bins - 1] = 0;
    for (let i = this.bins; i < n; i++) {
      re[i] = re[n - i];
      im[i] = -im[n - i];
    }
    this.fft.transform(re, im, true);
    const f = this.frames[j];
    for (let k = 0; k < n; k++) f[k] = re[k] * this.window[k] * this.olaScale;
  }

  private rotate() {
    this.mags.push(this.mags.shift()!);
    this.unitRe.push(this.unitRe.shift()!);
    this.unitIm.push(this.unitIm.shift()!);
    this.frames.push(this.frames.shift()!);
  }
}
