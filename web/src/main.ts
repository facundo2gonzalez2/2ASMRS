import "./style.css";
import type { PhaseMode } from "./dsp/framesynth";
import { SynthEngine } from "./engine/engine";
import type { FromWorklet } from "./engine/protocol";
import { compatible, fetchManifest, GROUP_LABELS, modelUrl, type Manifest, type ModelEntry } from "./models";
import { PRESET_COUNT, Store, type UiState } from "./state";
import { PresetBank } from "./ui/presets";
import { Spectrogram } from "./ui/spectrogram";
import { XYPad } from "./ui/xypad";

const COLOR_A = "#f6ad55";
const COLOR_B = "#9f7aea";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

const store = new Store();
let manifest: Manifest = { version: 1, archs: {}, models: [] };
let stats: FromWorklet | null = null;
let inferenceMs = 0;
let synthMs = 0;
let loadStatus = "";

// --- Motor ---

const spectrogram = new Spectrogram();
const engine = new SynthEngine({
  onFrame: (spectrum, decoderMs, phaseMs) => {
    spectrogram.push(spectrum);
    inferenceMs = 0.9 * inferenceMs + 0.1 * decoderMs;
    synthMs = 0.9 * synthMs + 0.1 * phaseMs;
  },
  onStats: (s) => {
    stats = s;
    renderMeter();
  },
  onError: (message) => showError(message),
});

// --- Helpers de modelos ---

const modelById = (id: string | null) => manifest.models.find((m) => m.id === id) ?? null;

function currentModels(): [ModelEntry, ModelEntry] | null {
  const a = modelById(store.state.modelA);
  const b = modelById(store.state.modelB);
  return a && b ? [a, b] : null;
}

/** Rango de cada dimensión: unión de los rangos de A y B. */
function zRanges(a: ModelEntry, b: ModelEntry): [number, number][] {
  return a.zRange.map(([lo, hi], i) => [Math.min(lo, b.zRange[i][0]), Math.max(hi, b.zRange[i][1])]);
}

function zInitAt(a: ModelEntry, b: ModelEntry, alpha: number): number[] {
  return a.zInit.map((v, i) => (1 - alpha) * v + alpha * b.zInit[i]);
}

function pickDefaultModels() {
  const models = manifest.models;
  let a = modelById(store.state.modelA) ?? models.find((m) => m.id.startsWith("piano")) ?? models[0] ?? null;
  let b = modelById(store.state.modelB);
  if (!b || !a || !compatible(a, b)) {
    b = models.find((m) => a && m.id !== a.id && compatible(a, m) && m.group === a.group) ?? a;
  }
  store.set({ modelA: a?.id ?? null, modelB: b?.id ?? null });
}

// --- DOM ---

const app = $("#app");
app.innerHTML = `
  <header>
    <div class="brand">
      <h1>2ASMRS</h1>
      <span class="subtitle">Sintetizador por autoencoder · interpolación de modelos en tiempo real</span>
    </div>
    <button id="play" class="play" type="button" aria-pressed="false"><span class="play-icon"></span><span class="play-label">Play</span></button>
  </header>

  <div id="error" class="error" hidden></div>
  <div id="resume" class="resume" hidden>Tocá cualquier parte para reanudar el audio</div>

  <section class="panel models">
    <label class="model a"><span class="tag">A</span><select id="model-a"></select></label>
    <button id="swap" class="icon-btn" type="button" title="Intercambiar A y B">⇄</button>
    <label class="model b"><span class="tag">B</span><select id="model-b"></select></label>
    <button id="reload" class="icon-btn" type="button" title="Releer models/manifest.json (modelos recién exportados)">↻</button>
    <div id="model-info" class="model-info"></div>
  </section>

  <section class="panel morph">
    <div class="row-label"><span>Interpolación α</span><output id="alpha-out"></output></div>
    <div class="alpha-row">
      <span class="end a" id="alpha-a">A</span>
      <input id="alpha" type="range" min="0" max="1" step="0.001" />
      <span class="end b" id="alpha-b">B</span>
    </div>
  </section>

  <section class="panel pads">
    <div id="pads" class="pad-grid"></div>
    <div id="extra-z" class="extra-z"></div>
  </section>

  <section class="panel output">
    <div class="volume">
      <div class="row-label"><span>Volumen</span><output id="volume-out"></output></div>
      <input id="volume" type="range" min="-60" max="6" step="0.5" />
      <div class="meter"><div id="meter-rms"></div><div id="meter-peak"></div></div>
    </div>
    <div class="preset-wrap">
      <div class="row-label"><span>Presets</span><span class="hint">click: cargar · mantener: guardar · teclas 1–0 / Shift+n</span></div>
      <div id="presets"></div>
    </div>
  </section>

  <section class="panel spec">
    <div id="spec"></div>
    <div id="status" class="status"></div>
  </section>

  <details class="panel advanced">
    <summary>Avanzado</summary>
    <div class="adv-grid">
      <label>Fase
        <select id="phase">
          <option value="rtisi">RTISI-LA: Griffin-Lim en tiempo real (mejor calidad)</option>
          <option value="pghi">RTPGHI: sin latencia extra</option>
          <option value="random">Aleatoria (como el sintetizador C++)</option>
        </select>
      </label>
      <label class="rtisi-only">Lookahead RTISI <output id="lookahead-out"></output>
        <input id="lookahead" type="range" min="0" max="3" step="1" />
      </label>
      <label class="rtisi-only">Iteraciones RTISI <output id="iterations-out"></output>
        <input id="iterations" type="range" min="1" max="16" step="1" />
      </label>
      <label>Contraste (× Xmax) <output id="contrast-out"></output>
        <input id="contrast" type="range" min="0.5" max="2" step="0.05" />
      </label>
      <label>Glide <output id="glide-out"></output>
        <input id="glide" type="range" min="0" max="500" step="5" />
      </label>
      <label>Buffer <output id="latency-out"></output>
        <input id="latency" type="range" min="1" max="12" step="1" />
      </label>
      <label class="check"><input id="autogain" type="checkbox" /> Auto-ganancia (normaliza a −20 dB RMS)</label>
      <div class="adv-buttons">
        <button id="reset-z" type="button">Z al inicial de A/B</button>
        <button id="reset-all" type="button">Restablecer todo (conserva presets)</button>
      </div>
    </div>
  </details>
`;

$("#spec").appendChild(spectrogram.el);

const selA = $<HTMLSelectElement>("#model-a");
const selB = $<HTMLSelectElement>("#model-b");
const alphaInput = $<HTMLInputElement>("#alpha");
const volumeInput = $<HTMLInputElement>("#volume");
const phaseSelect = $<HTMLSelectElement>("#phase");
const contrastInput = $<HTMLInputElement>("#contrast");
const lookaheadInput = $<HTMLInputElement>("#lookahead");
const iterationsInput = $<HTMLInputElement>("#iterations");
const glideInput = $<HTMLInputElement>("#glide");
const latencyInput = $<HTMLInputElement>("#latency");
const autoGainInput = $<HTMLInputElement>("#autogain");
const playBtn = $<HTMLButtonElement>("#play");

const setZ = (dim: number, value: number) => {
  const z = store.state.z.slice();
  z[dim] = value;
  store.set({ z, activePreset: null });
};

const pads = [
  new XYPad({
    title: "Z0 · Z1",
    xLabel: "Z0",
    yLabel: "Z1",
    onChange: (x, y) => {
      const z = store.state.z.slice();
      z[0] = x;
      z[1] = y;
      store.set({ z, activePreset: null });
    },
    onReset: () => resetZ([0, 1]),
  }),
  new XYPad({
    title: "Z2 · Z3",
    xLabel: "Z2",
    yLabel: "Z3",
    onChange: (x, y) => {
      const z = store.state.z.slice();
      z[2] = x;
      if (z.length > 3) z[3] = y;
      store.set({ z, activePreset: null });
    },
    onReset: () => resetZ([2, 3]),
  }),
];
pads.forEach((p) => $("#pads").appendChild(p.el));

function resetZ(dims?: number[]) {
  const pair = currentModels();
  if (!pair) return;
  const init = zInitAt(pair[0], pair[1], store.state.alpha);
  const z = store.state.z.slice();
  for (const d of dims ?? init.map((_, i) => i)) if (d < init.length) z[d] = init[d];
  store.set({ z, activePreset: null });
}

const presetBank = new PresetBank({
  count: PRESET_COUNT,
  onRecall: (i) => {
    const p = store.state.presets[i];
    if (!p) return;
    store.set({ z: p.z.slice(), alpha: p.alpha, volumeDb: p.volumeDb, activePreset: i });
  },
  onSave: (i) => {
    const s = store.state;
    const presets = s.presets.slice();
    presets[i] = { z: s.z.slice(), alpha: s.alpha, volumeDb: s.volumeDb, savedAt: new Date().toISOString() };
    store.set({ presets, activePreset: i });
  },
});
$("#presets").appendChild(presetBank.el);

// --- Eventos de la UI ---

selA.addEventListener("change", () => onModelPick({ modelA: selA.value }));
selB.addEventListener("change", () => onModelPick({ modelB: selB.value }));
$("#swap").addEventListener("click", () => {
  const { modelA, modelB, alpha } = store.state;
  store.set({ modelA: modelB, modelB: modelA, alpha: 1 - alpha });
});
$("#reload").addEventListener("click", () => void loadManifest());
alphaInput.addEventListener("input", () => store.set({ alpha: Number(alphaInput.value), activePreset: null }));
alphaInput.addEventListener("dblclick", () => store.set({ alpha: 0.5 }));
volumeInput.addEventListener("input", () => store.set({ volumeDb: Number(volumeInput.value), activePreset: null }));
phaseSelect.addEventListener("change", () => store.set({ phaseMode: phaseSelect.value as PhaseMode }));
contrastInput.addEventListener("input", () => store.set({ contrast: Number(contrastInput.value) }));
lookaheadInput.addEventListener("input", () => store.set({ rtisiLookahead: Number(lookaheadInput.value) }));
iterationsInput.addEventListener("input", () => store.set({ rtisiIterations: Number(iterationsInput.value) }));
glideInput.addEventListener("input", () => store.set({ glideMs: Number(glideInput.value) }));
latencyInput.addEventListener("input", () => store.set({ latencyBlocks: Number(latencyInput.value) }));
autoGainInput.addEventListener("change", () => store.set({ autoGain: autoGainInput.checked }));
$("#reset-z").addEventListener("click", () => resetZ());
$("#reset-all").addEventListener("click", () => {
  store.reset();
  pickDefaultModels();
  resetZ();
});
playBtn.addEventListener("click", () => void togglePlay(!store.state.playing));
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !(e.target as HTMLElement).closest("input, select, textarea, button, canvas")) {
    e.preventDefault();
    void togglePlay(!store.state.playing);
  }
});

/** Al elegir un modelo B incompatible con A, se ajusta el otro; si Z queda fuera del rango nuevo, vuelve al inicial. */
function onModelPick(patch: Partial<UiState>) {
  const a = modelById(patch.modelA ?? store.state.modelA);
  let b = modelById(patch.modelB ?? store.state.modelB);
  if (a && (!b || !compatible(a, b))) b = manifest.models.find((m) => compatible(a, m)) ?? a;
  store.set({ modelA: a?.id ?? null, modelB: b?.id ?? null });
  if (!a || !b) return;
  const ranges = zRanges(a, b);
  const outside = store.state.z.some((v, i) => i < ranges.length && (v < ranges[i][0] || v > ranges[i][1]));
  if (outside || store.state.z.length !== a.latentDim) resetZ();
}

async function togglePlay(playing: boolean) {
  store.set({ playing });
  $("#resume").hidden = true;
  try {
    await engine.setPlaying(playing);
  } catch (e) {
    showError(`No se pudo iniciar el audio: ${e instanceof Error ? e.message : e}`);
    store.set({ playing: false });
  }
}

// --- Estado → UI y motor ---

let loadedPair = "";

store.subscribe((s, changed) => {
  if (changed.has("modelA") || changed.has("modelB")) void syncModels();
  if (changed.has("z") || changed.has("modelA") || changed.has("modelB") || changed.has("alpha")) renderPads();
  if (changed.has("presets") || changed.has("activePreset")) renderPresets();
  if (changed.has("volumeDb") || changed.has("autoGain") || changed.has("latencyBlocks")) {
    engine.configure({ volumeDb: s.volumeDb, autoGain: s.autoGain, targetBlocks: s.latencyBlocks });
  }
  renderControls();
  pushParams();
});

function pushParams() {
  const s = store.state;
  engine.setParams({
    z: s.z,
    alpha: s.alpha,
    contrast: s.contrast,
    phaseMode: s.phaseMode,
    rtisiLookahead: s.rtisiLookahead,
    rtisiIterations: s.rtisiIterations,
    glideMs: s.glideMs,
  });
}

async function syncModels() {
  const pair = currentModels();
  renderModelSelects();
  if (!pair) return;
  const [a, b] = pair;
  const key = `${a.id}|${a.sha256}|${b.id}|${b.sha256}`;
  if (key === loadedPair) return;
  loadedPair = key;
  spectrogram.configure(a.sampleRate, a.winLength / 2 + 1);
  loadStatus = "cargando modelos…";
  renderStatus();
  try {
    const ms = await engine.load({
      graphUrl: modelUrl(manifest.archs[a.arch].graph),
      a: { id: a.id, weightsUrl: modelUrl(a.weights, a.sha256), xMax: a.xMax, dbMinNorm: a.dbMinNorm },
      b: { id: b.id, weightsUrl: modelUrl(b.weights, b.sha256), xMax: b.xMax, dbMinNorm: b.dbMinNorm },
      latentDim: a.latentDim,
      sampleRate: a.sampleRate,
      winLength: a.winLength,
      hopLength: a.hopLength,
    });
    if (loadedPair === key) loadStatus = `sesión ONNX en ${ms.toFixed(0)} ms`;
  } catch (e) {
    loadedPair = "";
    loadStatus = "error al cargar";
    showError(`No se pudieron cargar los modelos: ${e instanceof Error ? e.message : e}`);
  }
  renderStatus();
}

function renderModelSelects() {
  const a = modelById(store.state.modelA);
  for (const [sel, value] of [
    [selA, store.state.modelA],
    [selB, store.state.modelB],
  ] as const) {
    const groups = new Map<string, ModelEntry[]>();
    for (const m of manifest.models) groups.set(m.group, [...(groups.get(m.group) ?? []), m]);
    sel.replaceChildren(
      ...[...groups].map(([group, models]) => {
        const og = document.createElement("optgroup");
        og.label = GROUP_LABELS[group] ?? group;
        for (const m of models) {
          const opt = new Option(m.label, m.id, false, m.id === value);
          if (sel === selB && a && !compatible(a, m)) {
            opt.disabled = true;
            opt.label = `${m.label} (incompatible con A)`;
          }
          og.appendChild(opt);
        }
        return og;
      }),
    );
    sel.value = value ?? "";
  }
  const pair = currentModels();
  const info = $("#model-info");
  if (pair) {
    const [ma, mb] = pair;
    const describe = (m: ModelEntry) => `${m.beta > 0 ? `β=${m.beta}` : "sin β"} · Xmax ${m.xMax.toFixed(1)}`;
    info.innerHTML = `<span style="color:${COLOR_A}">A</span> ${describe(ma)} &nbsp; <span style="color:${COLOR_B}">B</span> ${describe(mb)} &nbsp; · ${ma.sampleRate} Hz · ventana ${ma.winLength} / hop ${ma.hopLength}`;
    $("#alpha-a").textContent = ma.label;
    $("#alpha-b").textContent = mb.label;
  } else {
    info.textContent = manifest.models.length ? "" : "No hay modelos exportados.";
  }
}

function renderPads() {
  const pair = currentModels();
  if (!pair) return;
  const [a, b] = pair;
  const ranges = zRanges(a, b);
  const z = store.state.z;
  const dim = a.latentDim;
  pads[0].setRanges(ranges[0], dim > 1 ? ranges[1] : null);
  pads[0].setValue(z[0] ?? 0, z[1] ?? 0);
  pads[0].setMarkers([
    { x: a.zInit[0], y: a.zInit[1] ?? 0, label: "A", color: COLOR_A },
    { x: b.zInit[0], y: b.zInit[1] ?? 0, label: "B", color: COLOR_B },
  ]);
  pads[1].el.hidden = dim < 3;
  if (dim >= 3) {
    pads[1].setRanges(ranges[2], dim > 3 ? ranges[3] : null);
    pads[1].setValue(z[2] ?? 0, z[3] ?? 0);
    pads[1].setMarkers([
      { x: a.zInit[2], y: a.zInit[3] ?? 0, label: "A", color: COLOR_A },
      { x: b.zInit[2], y: b.zInit[3] ?? 0, label: "B", color: COLOR_B },
    ]);
  }
  renderExtraZ(dim, ranges);
}

/** Dimensiones latentes más allá de Z3 (modelos con latent_dim > 4) como sliders. */
function renderExtraZ(dim: number, ranges: [number, number][]) {
  const box = $("#extra-z");
  const wanted = Math.max(0, dim - 4);
  if (box.children.length !== wanted) {
    box.replaceChildren(
      ...Array.from({ length: wanted }, (_, k) => {
        const d = k + 4;
        const label = document.createElement("label");
        label.innerHTML = `<span>Z${d}</span><input type="range" step="0.001" /><output></output>`;
        label.querySelector("input")!.addEventListener("input", (e) => setZ(d, Number((e.target as HTMLInputElement).value)));
        return label;
      }),
    );
  }
  [...box.children].forEach((label, k) => {
    const d = k + 4;
    const input = label.querySelector("input")!;
    input.min = String(ranges[d][0]);
    input.max = String(ranges[d][1]);
    input.value = String(store.state.z[d] ?? 0);
    label.querySelector("output")!.textContent = (store.state.z[d] ?? 0).toFixed(2);
  });
}

function renderPresets() {
  const s = store.state;
  presetBank.update(
    s.presets.map((p) => p !== null),
    s.activePreset,
    s.presets.map((p, i) =>
      p
        ? `Preset ${i + 1}: α ${p.alpha.toFixed(2)} · Z [${p.z.map((v) => v.toFixed(2)).join(", ")}] · ${p.volumeDb} dB\n` +
          `Guardado ${new Date(p.savedAt).toLocaleString()}. Mantener apretado para sobrescribir.`
        : `Preset ${i + 1}: vacío. Mantener apretado para guardar.`,
    ),
  );
}

function renderControls() {
  const s = store.state;
  alphaInput.value = String(s.alpha);
  $("#alpha-out").textContent = s.alpha.toFixed(3);
  alphaInput.style.setProperty("--fill", `${s.alpha * 100}%`);
  volumeInput.value = String(s.volumeDb);
  $("#volume-out").textContent = s.volumeDb <= -60 ? "mute" : `${s.volumeDb.toFixed(1)} dB`;
  phaseSelect.value = s.phaseMode;
  document.querySelectorAll<HTMLElement>(".rtisi-only").forEach((el) => (el.hidden = s.phaseMode !== "rtisi"));
  lookaheadInput.value = String(s.rtisiLookahead);
  iterationsInput.value = String(s.rtisiIterations);
  $("#iterations-out").textContent = String(s.rtisiIterations);
  contrastInput.value = String(s.contrast);
  $("#contrast-out").textContent = `${s.contrast.toFixed(2)}×`;
  glideInput.value = String(s.glideMs);
  $("#glide-out").textContent = `${s.glideMs} ms`;
  latencyInput.value = String(s.latencyBlocks);
  const hopMs = currentHopMs();
  $("#lookahead-out").textContent = `${s.rtisiLookahead} frames = +${(s.rtisiLookahead * hopMs).toFixed(0)} ms`;
  $("#latency-out").textContent = `${s.latencyBlocks} bloques ≈ ${(s.latencyBlocks * hopMs).toFixed(0)} ms`;
  autoGainInput.checked = s.autoGain;
  playBtn.classList.toggle("on", s.playing);
  playBtn.setAttribute("aria-pressed", String(s.playing));
  playBtn.querySelector(".play-label")!.textContent = s.playing ? "Stop" : "Play";
}

function currentHopMs() {
  const pair = currentModels();
  return pair ? (1000 * pair[0].hopLength) / pair[0].sampleRate : 23.2;
}

function renderMeter() {
  if (!stats) return;
  const pct = (db: number) => `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
  const out = stats.rmsDb + (store.state.autoGain ? stats.agcDb : 0) + store.state.volumeDb;
  $("#meter-rms").style.width = store.state.playing ? pct(out) : "0%";
  $("#meter-peak").style.left = store.state.playing ? pct(20 * Math.log10(stats.peak + 1e-9)) : "0%";
}

function renderStatus() {
  const parts = [loadStatus];
  const sr = engine.audioSampleRate;
  if (sr) parts.push(`${sr} Hz`);
  if (inferenceMs) parts.push(`decoder ${inferenceMs.toFixed(1)} + fase ${synthMs.toFixed(1)} ms/frame`);
  if (stats && store.state.playing) {
    const hopMs = currentHopMs();
    const lookahead = store.state.phaseMode === "rtisi" ? store.state.rtisiLookahead : 0;
    parts.push(`latencia ≈ ${((stats.queued + lookahead) * hopMs + engine.outputLatencyMs).toFixed(0)} ms`);
    if (stats.underruns) parts.push(`cortes ${stats.underruns}`);
    if (store.state.autoGain) parts.push(`AGC ${stats.agcDb >= 0 ? "+" : ""}${stats.agcDb.toFixed(0)} dB`);
  }
  $("#status").textContent = parts.filter(Boolean).join(" · ");
}
window.setInterval(renderStatus, 500);

function showError(message: string) {
  const el = $("#error");
  el.textContent = message;
  el.hidden = false;
  window.clearTimeout((showError as unknown as { t?: number }).t);
  (showError as unknown as { t?: number }).t = window.setTimeout(() => (el.hidden = true), 8000);
}

async function loadManifest() {
  try {
    manifest = await fetchManifest();
  } catch (e) {
    showError(e instanceof Error ? e.message : String(e));
    return;
  }
  pickDefaultModels();
  loadedPair = "";
  await syncModels();
  const pair = currentModels();
  if (pair && store.state.z.length !== pair[0].latentDim) resetZ();
  renderPads();
}

// --- Arranque ---

const s0 = store.state;
engine.configure({ volumeDb: s0.volumeDb, autoGain: s0.autoGain, targetBlocks: s0.latencyBlocks, playing: false });
renderControls();
renderPresets();
pushParams();
await loadManifest();

// El browser no deja arrancar audio sin un gesto: si estaba sonando, se reanuda con la primera interacción.
if (s0.playing) {
  store.set({ playing: false });
  $("#resume").hidden = false;
  const resume = (e: Event) => {
    window.removeEventListener("pointerdown", resume, true);
    window.removeEventListener("keydown", resume, true);
    // El botón Play y la barra espaciadora ya hacen toggle por su cuenta
    const own = (e.target as HTMLElement).closest("#play") || (e instanceof KeyboardEvent && e.code === "Space");
    if (!own) void togglePlay(true);
  };
  window.addEventListener("pointerdown", resume, true);
  window.addEventListener("keydown", resume, true);
}
