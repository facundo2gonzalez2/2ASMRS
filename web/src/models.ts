/** Manifest de modelos exportados por model/scripts/export_web_models.py (public/models/manifest.json). */

export interface ArchInfo {
  graph: string;
  layers: number[];
  externalData: string[];
  weightsBytes: number;
}

export interface ModelEntry {
  id: string;
  label: string;
  group: string;
  arch: string;
  weights: string;
  sha256: string;
  latentDim: number;
  sampleRate: number;
  winLength: number;
  hopLength: number;
  xMax: number;
  dbMinNorm: number;
  beta: number;
  zRange: [number, number][];
  zInit: number[];
  source: string;
  exportedAt: string;
}

export interface Manifest {
  version: number;
  archs: Record<string, ArchInfo>;
  models: ModelEntry[];
}

const MODELS_BASE = new URL("models/", new URL(import.meta.env.BASE_URL, window.location.href));

export async function fetchManifest(): Promise<Manifest> {
  const res = await fetch(new URL("manifest.json", MODELS_BASE), { cache: "no-cache" });
  if (!res.ok) throw new Error(`No se pudo leer models/manifest.json (${res.status}). ¿Corriste export_web_models.py?`);
  return res.json();
}

export function modelUrl(path: string, version?: string): string {
  const url = new URL(path, MODELS_BASE);
  if (version) url.searchParams.set("v", version);
  return url.href;
}

/** Dos modelos se pueden interpolar si comparten arquitectura y parámetros de STFT. */
export function compatible(a: ModelEntry, b: ModelEntry): boolean {
  return (
    a.arch === b.arch &&
    a.sampleRate === b.sampleRate &&
    a.winLength === b.winLength &&
    a.hopLength === b.hopLength
  );
}

export const GROUP_LABELS: Record<string, string> = {
  instruments_from_checkpoint: "Instrumentos (fine-tuning desde base)",
  instruments_from_scratch: "Instrumentos (desde cero)",
  base_model: "Modelo base",
};
