/** Espectrograma en cascada de la salida del decoder (eje de frecuencia logarítmico). */

const MIN_HZ = 40;

export class Spectrogram {
  readonly el: HTMLCanvasElement;
  private pending: Float32Array[] = [];
  private rowBins: Int32Array = new Int32Array(0);
  private sampleRate = 22050;
  private bins = 1025;
  private column: ImageData | null = null;
  private readonly palette: Uint8ClampedArray;

  constructor() {
    this.el = document.createElement("canvas");
    this.el.className = "spectrogram";
    this.palette = makePalette();
    new ResizeObserver(() => this.resize()).observe(this.el);
    const loop = () => {
      this.draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  configure(sampleRate: number, bins: number) {
    if (sampleRate === this.sampleRate && bins === this.bins) return;
    this.sampleRate = sampleRate;
    this.bins = bins;
    this.resize();
  }

  push(spectrum: Float32Array) {
    this.pending.push(spectrum);
    if (this.pending.length > 64) this.pending.shift();
  }

  private resize() {
    const c = this.el;
    const h = Math.max(1, Math.round(c.clientHeight));
    const w = Math.max(1, Math.round(c.clientWidth));
    if (c.width === w && c.height === h && this.rowBins.length === h) return;
    c.width = w;
    c.height = h;
    const nyq = this.sampleRate / 2;
    this.rowBins = new Int32Array(h);
    for (let r = 0; r < h; r++) {
      const frac = 1 - r / (h - 1 || 1); // fila 0 = arriba = agudos
      const hz = MIN_HZ * Math.pow(nyq / MIN_HZ, frac);
      this.rowBins[r] = Math.min(this.bins - 1, Math.round((hz / nyq) * (this.bins - 1)));
    }
    this.column = c.getContext("2d")!.createImageData(1, h);
    const g = c.getContext("2d")!;
    g.fillStyle = "#07080c";
    g.fillRect(0, 0, w, h);
  }

  private draw() {
    if (!this.pending.length || !this.column) return;
    const c = this.el;
    const g = c.getContext("2d")!;
    const n = this.pending.length;
    const step = 2; // px por frame
    g.drawImage(c, -n * step, 0);
    const data = this.column.data;
    for (let k = 0; k < n; k++) {
      const s = this.pending[k];
      for (let r = 0; r < this.rowBins.length; r++) {
        const v = s[this.rowBins[r]];
        const idx = Math.max(0, Math.min(255, Math.round((Number.isFinite(v) ? v : 0) * 255))) * 3;
        data[r * 4] = this.palette[idx];
        data[r * 4 + 1] = this.palette[idx + 1];
        data[r * 4 + 2] = this.palette[idx + 2];
        data[r * 4 + 3] = 255;
      }
      for (let p = 0; p < step; p++) g.putImageData(this.column, c.width - (n - k) * step + p, 0);
    }
    this.pending = [];
  }
}

/** Mapa de color oscuro → turquesa → amarillo. */
function makePalette(): Uint8ClampedArray {
  const stops: [number, [number, number, number]][] = [
    [0, [7, 8, 12]],
    [0.35, [20, 40, 80]],
    [0.6, [32, 140, 150]],
    [0.8, [120, 210, 160]],
    [1, [250, 240, 160]],
  ];
  const p = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let j = 0;
    while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
    const [t0, c0] = stops[j];
    const [t1, c1] = stops[j + 1];
    const u = (t - t0) / (t1 - t0);
    for (let ch = 0; ch < 3; ch++) p[i * 3 + ch] = c0[ch] + u * (c1[ch] - c0[ch]);
  }
  return p;
}
