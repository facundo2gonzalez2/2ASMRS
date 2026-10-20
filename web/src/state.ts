/** Estado de la UI, persistido continuamente en localStorage. */
import type { PhaseMode } from "./dsp/framesynth";

export const PRESET_COUNT = 10;

export interface Preset {
  z: number[];
  alpha: number;
  volumeDb: number;
  savedAt: string;
}

export interface UiState {
  modelA: string | null;
  modelB: string | null;
  z: number[];
  alpha: number;
  volumeDb: number;
  playing: boolean;
  presets: (Preset | null)[];
  activePreset: number | null;
  // Avanzado
  phaseMode: PhaseMode;
  rtisiLookahead: number;
  rtisiIterations: number;
  contrast: number;
  glideMs: number;
  autoGain: boolean;
  latencyBlocks: number;
}

const STORAGE_KEY = "2asmrs.ui-state.v1";

export const DEFAULT_STATE: UiState = {
  modelA: "piano_from_checkpoint_beta_0.001",
  modelB: "guitar_from_checkpoint_beta_0.001",
  z: [], // vacío: al cargar los modelos se inicializa con el Z inicial de A
  alpha: 0,
  volumeDb: -12,
  playing: false,
  presets: Array(PRESET_COUNT).fill(null),
  activePreset: null,
  phaseMode: "rtisi",
  rtisiLookahead: 2,
  rtisiIterations: 8,
  contrast: 1.45, // mismo default que la demo de Streamlit
  glideMs: 40,
  autoGain: true,
  latencyBlocks: 3,
};

type Listener = (state: UiState, changed: Set<keyof UiState>) => void;

export class Store {
  private value: UiState;
  private readonly listeners = new Set<Listener>();
  private saveTimer: number | undefined;

  constructor() {
    this.value = load();
    // Por si se cierra la pestaña dentro de la ventana del debounce
    window.addEventListener("pagehide", () => this.flush());
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && this.flush());
  }

  get state(): Readonly<UiState> {
    return this.value;
  }

  set(patch: Partial<UiState>) {
    const changed = new Set<keyof UiState>();
    for (const key of Object.keys(patch) as (keyof UiState)[]) {
      if (patch[key] !== this.value[key]) changed.add(key);
    }
    if (changed.size === 0) return;
    this.value = { ...this.value, ...patch };
    for (const listener of this.listeners) listener(this.value, changed);
    this.scheduleSave();
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  reset() {
    this.set({ ...DEFAULT_STATE, presets: this.value.presets });
  }

  private scheduleSave() {
    if (this.saveTimer !== undefined) return;
    this.saveTimer = window.setTimeout(() => this.flush(), 150);
  }

  private flush() {
    window.clearTimeout(this.saveTimer);
    this.saveTimer = undefined;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.value));
    } catch {
      // localStorage lleno o deshabilitado: la UI sigue funcionando sin persistencia
    }
  }
}

function load(): UiState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULT_STATE);
    const saved = JSON.parse(raw) as Partial<UiState>;
    const state = { ...structuredClone(DEFAULT_STATE), ...saved };
    state.presets = Array.from({ length: PRESET_COUNT }, (_, i) => saved.presets?.[i] ?? null);
    return state;
  } catch {
    return structuredClone(DEFAULT_STATE);
  }
}
