/**
 * Reconstrucción de fase frame a frame a partir de magnitudes STFT.
 *
 * - `RtPghi`: Real-Time Phase Gradient Heap Integration causal (Průša & Holighaus, 2017), la versión en tiempo
 *   real del PGHI que se usa offline (tifresi). Las derivadas de la fase salen del log-magnitud aproximando la
 *   ventana Hann por una gaussiana (λ = 0.25645·N²): derivada temporal desde la pendiente en frecuencia del frame
 *   actual, derivada en frecuencia desde la diferencia con el frame anterior (sin lookahead). Para un Z quieto da
 *   un sonido tonal estable (las fases quedan enganchadas alrededor de cada pico, como un phase vocoder con
 *   phase locking).
 * - `randomPhase`: fase uniforme independiente por frame, como el sintetizador C++ (textura ruidosa).
 *
 * Convención de fase "frequency-invariant": referida al centro de la ventana. Para el iFFT del frame hay que
 * multiplicar por (−1)^m (ver FrameSynth).
 */

const HANN_LAMBDA = 0.25645;
const TWO_PI = 2 * Math.PI;

export class RtPghi {
  readonly bins: number;
  private readonly n: number;
  private readonly hop: number;
  private readonly lambda: number;
  private readonly logTol: number;
  private sPrev: Float64Array;
  private sCur: Float64Array;
  private tgPrev: Float64Array; // derivada temporal de la fase (rad/muestra)
  private tgCur: Float64Array;
  private readonly fg: Float64Array; // derivada en frecuencia (rad por ciclo/muestra)
  private phasePrev: Float64Array;
  private phaseCur: Float64Array;
  private readonly done: Uint8Array;
  private readonly heap: Int32Array;
  private heapSize = 0;
  private first = true;

  /** @param tol bins por debajo de max·tol reciben fase aleatoria (no participan de la integración) */
  constructor(winLength: number, hopLength: number, tol = 1e-5) {
    this.n = winLength;
    this.hop = hopLength;
    this.bins = winLength / 2 + 1;
    this.lambda = HANN_LAMBDA * winLength * winLength;
    this.logTol = Math.log(tol);
    const m = this.bins;
    this.sPrev = new Float64Array(m);
    this.sCur = new Float64Array(m);
    this.tgPrev = new Float64Array(m);
    this.tgCur = new Float64Array(m);
    this.fg = new Float64Array(m);
    this.phasePrev = new Float64Array(m);
    this.phaseCur = new Float64Array(m);
    this.done = new Uint8Array(m);
    this.heap = new Int32Array(2 * m);
    this.reset();
  }

  reset(): void {
    this.first = true;
    for (let i = 0; i < this.bins; i++) this.phasePrev[i] = Math.random() * TWO_PI;
  }

  /**
   * Calcula la fase del siguiente frame.
   * @param logMag log natural de la magnitud (largo `bins`)
   * @returns fases del frame (vista interna, válida hasta la próxima llamada)
   */
  next(logMag: Float32Array | Float64Array): Float64Array {
    const m = this.bins;
    const n = this.n;
    const hop = this.hop;

    // Rotar buffers: cur → prev
    [this.sPrev, this.sCur] = [this.sCur, this.sPrev];
    [this.tgPrev, this.tgCur] = [this.tgCur, this.tgPrev];
    [this.phasePrev, this.phaseCur] = [this.phaseCur, this.phasePrev];
    if (this.first) {
      // phasePrev quedó con las fases aleatorias de reset() (ahora en phaseCur por la rotación)
      [this.phasePrev, this.phaseCur] = [this.phaseCur, this.phasePrev];
    }

    const s = this.sCur;
    let sMax = -Infinity;
    for (let i = 0; i < m; i++) {
      s[i] = logMag[i];
      if (s[i] > sMax) sMax = s[i];
    }
    if (this.first) {
      // Sin historia: el frame anterior se toma igual al actual (derivadas en frecuencia nulas)
      this.sPrev.set(s);
    }

    // Derivadas: φ_t = (1/λ)·∂s/∂ω + 2πω ; φ_ω = −λ·∂s/∂t   (ω en ciclos/muestra, t en muestras)
    const tg = this.tgCur;
    const tScale = n / (2 * this.lambda);
    for (let i = 0; i < m; i++) {
      const ds = i === 0 || i === m - 1 ? 0 : s[i + 1] - s[i - 1];
      tg[i] = tScale * ds + (TWO_PI * i) / n;
      this.fg[i] = (-this.lambda * (s[i] - this.sPrev[i])) / hop;
    }
    if (this.first) this.tgPrev.set(tg);

    const thresh = sMax + this.logTol;
    const phase = this.phaseCur;
    const phasePrev = this.phasePrev;
    const done = this.done;
    let remaining = 0;
    for (let i = 0; i < m; i++) {
      if (s[i] < thresh) {
        done[i] = 1;
        phase[i] = Math.random() * TWO_PI;
      } else {
        done[i] = 0;
        remaining++;
      }
    }

    // Heap: índices < m son bins del frame anterior, >= m del actual
    this.heapSize = 0;
    for (let i = 0; i < m; i++) if (this.sPrev[i] >= thresh) this.push(i);

    const halfHop = hop / 2;
    const freqStep = 1 / (2 * n);
    while (remaining > 0) {
      if (this.heapSize === 0) {
        // Bin significativo sin vecinos resueltos (onset): continuar en el tiempo desde el frame anterior
        let best = -1;
        for (let i = 0; i < m; i++) if (!done[i] && (best < 0 || s[i] > s[best])) best = i;
        phase[best] = phasePrev[best] + halfHop * (this.tgPrev[best] + tg[best]);
        done[best] = 1;
        remaining--;
        this.push(best + m);
        continue;
      }
      const top = this.pop();
      if (top < m) {
        if (!done[top]) {
          phase[top] = phasePrev[top] + halfHop * (this.tgPrev[top] + tg[top]);
          done[top] = 1;
          remaining--;
          this.push(top + m);
        }
      } else {
        const i = top - m;
        if (i + 1 < m && !done[i + 1]) {
          phase[i + 1] = phase[i] + freqStep * (this.fg[i] + this.fg[i + 1]);
          done[i + 1] = 1;
          remaining--;
          this.push(i + 1 + m);
        }
        if (i - 1 >= 0 && !done[i - 1]) {
          phase[i - 1] = phase[i] - freqStep * (this.fg[i] + this.fg[i - 1]);
          done[i - 1] = 1;
          remaining--;
          this.push(i - 1 + m);
        }
      }
    }

    // Mantener las fases acotadas para no perder precisión al acumular
    for (let i = 0; i < m; i++) phase[i] -= TWO_PI * Math.floor(phase[i] / TWO_PI);
    this.first = false;
    return phase;
  }

  /**
   * Reemplaza la fase del último frame calculado (la que se usa como "anterior" en el próximo `next`), p. ej. con
   * las fases refinadas por RTISI-LA. Convención frequency-invariant.
   */
  overrideLastPhase(phase: Float64Array): void {
    if (!this.first) this.phaseCur.set(phase);
  }

  private key(idx: number): number {
    return idx < this.bins ? this.sPrev[idx] : this.sCur[idx - this.bins];
  }

  private push(idx: number): void {
    const h = this.heap;
    let i = this.heapSize++;
    const k = this.key(idx);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.key(h[parent]) >= k) break;
      h[i] = h[parent];
      i = parent;
    }
    h[i] = idx;
  }

  private pop(): number {
    const h = this.heap;
    const top = h[0];
    const last = h[--this.heapSize];
    const k = this.key(last);
    let i = 0;
    const size = this.heapSize;
    while (true) {
      let child = 2 * i + 1;
      if (child >= size) break;
      if (child + 1 < size && this.key(h[child + 1]) > this.key(h[child])) child++;
      if (this.key(h[child]) <= k) break;
      h[i] = h[child];
      i = child;
    }
    if (size > 0) h[i] = last;
    return top;
  }
}

export function randomPhase(out: Float64Array): Float64Array {
  for (let i = 0; i < out.length; i++) out[i] = Math.random() * TWO_PI;
  return out;
}
