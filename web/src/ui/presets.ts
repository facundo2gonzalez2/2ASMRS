/** Banco de presets: click recupera, mantener apretado sobrescribe. Teclado: 1–9/0 recupera, Shift+número guarda. */

const HOLD_MS = 650;

export interface PresetBankOptions {
  count: number;
  onRecall: (index: number) => void;
  onSave: (index: number) => void;
}

export class PresetBank {
  readonly el: HTMLElement;
  private readonly buttons: HTMLButtonElement[] = [];
  private filled: boolean[] = [];

  constructor(opts: PresetBankOptions) {
    this.el = document.createElement("div");
    this.el.className = "presets";
    this.el.style.setProperty("--hold-ms", `${HOLD_MS}ms`);
    for (let i = 0; i < opts.count; i++) {
      const b = document.createElement("button");
      b.className = "preset";
      b.type = "button";
      b.innerHTML = `<span class="preset-num">${i + 1}</span><span class="preset-fill"></span>`;
      let timer: number | undefined;
      let saved = false;
      const cancel = () => {
        window.clearTimeout(timer);
        timer = undefined;
        b.classList.remove("holding");
      };
      b.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        saved = false;
        b.classList.add("holding");
        b.setPointerCapture(e.pointerId);
        timer = window.setTimeout(() => {
          saved = true;
          cancel();
          opts.onSave(i);
          this.flash(i, "saved");
        }, HOLD_MS);
      });
      b.addEventListener("pointerup", () => {
        const wasHolding = timer !== undefined;
        cancel();
        if (wasHolding && !saved) {
          if (this.filled[i]) opts.onRecall(i);
          else this.flash(i, "empty");
        }
      });
      b.addEventListener("pointercancel", cancel);
      b.addEventListener("contextmenu", (e) => e.preventDefault()); // long-press en mobile
      b.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          if (e.shiftKey) { opts.onSave(i); this.flash(i, "saved"); }
          else if (this.filled[i]) opts.onRecall(i);
        }
      });
      this.buttons.push(b);
      this.el.appendChild(b);
    }

    window.addEventListener("keydown", (e) => {
      const target = e.target as HTMLElement;
      if (target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
      const m = /^Digit(\d)$/.exec(e.code);
      if (!m) return;
      const i = (Number(m[1]) + 9) % 10; // 1→0 … 0→9
      if (i >= opts.count) return;
      e.preventDefault();
      if (e.shiftKey) { opts.onSave(i); this.flash(i, "saved"); }
      else if (this.filled[i]) opts.onRecall(i);
      else this.flash(i, "empty");
    });
  }

  update(filled: boolean[], active: number | null, titles: string[]) {
    this.filled = filled;
    this.buttons.forEach((b, i) => {
      b.classList.toggle("filled", filled[i]);
      b.classList.toggle("active", active === i);
      b.title = titles[i];
    });
  }

  /** Progreso (0–1) de la transición hacia el preset `index`; `null` la oculta. */
  setProgress(index: number | null, progress: number) {
    this.buttons.forEach((b, i) => {
      b.classList.toggle("morphing", i === index);
      if (i === index) b.style.setProperty("--morph-progress", String(progress));
    });
  }

  private flash(i: number, kind: "saved" | "empty") {
    const b = this.buttons[i];
    b.classList.remove("flash-saved", "flash-empty");
    void b.offsetWidth; // reiniciar la animación
    b.classList.add(`flash-${kind}`);
  }
}
