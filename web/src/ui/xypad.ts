/** Pad XY para dos dimensiones del espacio latente. Arrastre con mouse/touch, flechas con teclado, doble click resetea. */

export interface PadMarker {
  x: number;
  y: number;
  label: string;
  color: string;
}

export interface XYPadOptions {
  title: string;
  xLabel: string;
  yLabel: string;
  onChange: (x: number, y: number) => void;
  onReset?: () => void;
}

export class XYPad {
  readonly el: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly readout: HTMLElement;
  private readonly opts: XYPadOptions;
  private xRange: [number, number] = [-3, 3];
  private yRange: [number, number] | null = [-3, 3];
  private x = 0;
  private y = 0;
  private markers: PadMarker[] = [];
  private dragging = false;

  constructor(opts: XYPadOptions) {
    this.opts = opts;
    this.el = document.createElement("div");
    this.el.className = "xypad";
    this.el.innerHTML = `
      <div class="xypad-head"><span class="xypad-title"></span><span class="xypad-readout"></span></div>
      <canvas tabindex="0" role="slider" aria-roledescription="pad XY"></canvas>`;
    this.el.querySelector(".xypad-title")!.textContent = opts.title;
    this.readout = this.el.querySelector(".xypad-readout")!;
    this.canvas = this.el.querySelector("canvas")!;
    this.canvas.setAttribute("aria-label", `${opts.title}: ${opts.xLabel} / ${opts.yLabel}`);

    this.canvas.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      this.canvas.setPointerCapture(e.pointerId);
      this.canvas.focus();
      this.fromPointer(e);
    });
    this.canvas.addEventListener("pointermove", (e) => this.dragging && this.fromPointer(e));
    const end = () => (this.dragging = false);
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
    this.canvas.addEventListener("dblclick", () => opts.onReset?.());
    this.canvas.addEventListener("keydown", (e) => this.onKey(e));
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }

  setRanges(xRange: [number, number], yRange: [number, number] | null) {
    this.xRange = xRange;
    this.yRange = yRange;
    this.draw();
  }

  setValue(x: number, y: number) {
    this.x = x;
    this.y = y;
    this.draw();
  }

  setMarkers(markers: PadMarker[]) {
    this.markers = markers;
    this.draw();
  }

  private emit(x: number, y: number) {
    const [x0, x1] = this.xRange;
    this.x = Math.min(x1, Math.max(x0, x));
    if (this.yRange) this.y = Math.min(this.yRange[1], Math.max(this.yRange[0], y));
    this.draw();
    this.opts.onChange(this.x, this.y);
  }

  private fromPointer(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const u = (e.clientX - r.left) / r.width;
    const v = 1 - (e.clientY - r.top) / r.height;
    const [x0, x1] = this.xRange;
    const [y0, y1] = this.yRange ?? [0, 0];
    this.emit(x0 + u * (x1 - x0), this.yRange ? y0 + v * (y1 - y0) : this.y);
  }

  private onKey(e: KeyboardEvent) {
    const fine = e.shiftKey ? 0.002 : 0.02;
    const dx = (this.xRange[1] - this.xRange[0]) * fine;
    const dy = this.yRange ? (this.yRange[1] - this.yRange[0]) * fine : 0;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-dx, 0],
      ArrowRight: [dx, 0],
      ArrowUp: [0, dy],
      ArrowDown: [0, -dy],
    };
    const move = moves[e.key];
    if (!move) return;
    e.preventDefault();
    this.emit(this.x + move[0], this.y + move[1]);
  }

  private toPx(x: number, y: number, w: number, h: number): [number, number] {
    const [x0, x1] = this.xRange;
    const [y0, y1] = this.yRange ?? [-1, 1];
    return [((x - x0) / (x1 - x0)) * w, (1 - (this.yRange ? (y - y0) / (y1 - y0) : 0.5)) * h];
  }

  private draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(this.el);
    const grid = css.getPropertyValue("--pad-grid").trim() || "#2a2f3a";
    const axis = css.getPropertyValue("--pad-axis").trim() || "#465063";
    const accent = css.getPropertyValue("--accent").trim() || "#4fd1c5";
    g.clearRect(0, 0, w, h);

    g.strokeStyle = grid;
    g.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      const px = Math.round((i / 8) * w) + 0.5;
      const py = Math.round((i / 8) * h) + 0.5;
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
      if (this.yRange) { g.beginPath(); g.moveTo(0, py); g.lineTo(w, py); g.stroke(); }
    }
    // Ejes en 0
    const [ox, oy] = this.toPx(0, 0, w, h);
    g.strokeStyle = axis;
    if (ox > 0 && ox < w) { g.beginPath(); g.moveTo(ox, 0); g.lineTo(ox, h); g.stroke(); }
    if (this.yRange && oy > 0 && oy < h) { g.beginPath(); g.moveTo(0, oy); g.lineTo(w, oy); g.stroke(); }

    g.font = "11px system-ui, sans-serif";
    g.fillStyle = axis;
    g.fillText(`${this.opts.xLabel} →`, w - 48, h - 6);
    if (this.yRange) g.fillText(`↑ ${this.opts.yLabel}`, 6, 14);

    for (const m of this.markers) {
      const [mx, my] = this.toPx(m.x, m.y, w, h);
      g.strokeStyle = m.color;
      g.lineWidth = 1.5;
      g.beginPath(); g.arc(mx, my, 5, 0, 2 * Math.PI); g.stroke();
      g.fillStyle = m.color;
      g.fillText(m.label, mx + 8, my + 4);
    }

    const [px, py] = this.toPx(
      Math.min(this.xRange[1], Math.max(this.xRange[0], this.x)),
      this.yRange ? Math.min(this.yRange[1], Math.max(this.yRange[0], this.y)) : 0,
      w,
      h,
    );
    g.strokeStyle = accent;
    g.globalAlpha = 0.35;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
    if (this.yRange) { g.beginPath(); g.moveTo(0, py); g.lineTo(w, py); g.stroke(); }
    g.globalAlpha = 1;
    g.fillStyle = accent;
    g.beginPath(); g.arc(px, py, 9, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = "#0b0d12";
    g.lineWidth = 2;
    g.stroke();

    // Precisión según el rango (en los β-VAE algunas dimensiones colapsan a rangos de ±0.01)
    const span = Math.min(this.xRange[1] - this.xRange[0], this.yRange ? this.yRange[1] - this.yRange[0] : Infinity);
    const decimals = Math.min(5, Math.max(2, Math.ceil(-Math.log10(span / 100))));
    const fmt = (v: number) => (v >= 0 ? " " : "") + v.toFixed(decimals);
    this.readout.textContent = this.yRange
      ? `${this.opts.xLabel} ${fmt(this.x)}  ${this.opts.yLabel} ${fmt(this.y)}`
      : `${this.opts.xLabel} ${fmt(this.x)}`;
    this.canvas.setAttribute("aria-valuetext", this.readout.textContent);
  }
}
