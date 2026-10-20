/**
 * Paridad ONNX Runtime Web (wasm) vs PyTorch: el grafo interpolado con los pesos de dos checkpoints mapeados como
 * datos externos tiene que reproducir interpolar_vae(...).decoder(z) (fixtures/onnx_reference.json).
 * Requiere haber corrido model/scripts/export_web_models.py.
 */
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { InterpolatedDecoder } from "../src/engine/decoder";

const MODELS = new URL("../public/models/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", MODELS), "utf8"));
const cases: { a: string; b: string; alpha: number; z: number[][]; y: number[][] }[] = JSON.parse(
  readFileSync(new URL("./fixtures/onnx_reference.json", import.meta.url), "utf8"),
);

ort.env.wasm.numThreads = 1;

const bytes = (rel: string) => new Uint8Array(readFileSync(new URL(rel, MODELS)));
const model = (id: string) => manifest.models.find((m: { id: string }) => m.id === id);

let worst = 0;
const sessions = new Map<string, InterpolatedDecoder>();
for (const c of cases) {
  const key = `${c.a}|${c.b}`;
  if (!sessions.has(key)) {
    const ma = model(c.a);
    const mb = model(c.b);
    const graph = bytes(manifest.archs[ma.arch].graph);
    const t0 = performance.now();
    sessions.set(key, await InterpolatedDecoder.create(ort, graph, bytes(ma.weights), bytes(mb.weights), ma.latentDim));
    console.log(`     sesión ${key}: ${(performance.now() - t0).toFixed(0)} ms`);
  }
  const dec = sessions.get(key)!;
  for (let f = 0; f < c.z.length; f++) {
    const y = await dec.run(c.z[f], c.alpha);
    for (let i = 0; i < y.length; i++) worst = Math.max(worst, Math.abs(y[i] - c.y[f][i]));
  }
}
const ok = worst < 1e-4;
console.log(`${ok ? "ok  " : "FAIL"} onnx web vs pytorch: max abs err ${worst.toExponential(2)}`);
if (!ok) process.exitCode = 1;

const dec = [...sessions.values()][0];
const z = [0.1, 0.5, -0.2, 0.3];
const n = 300;
const t0 = performance.now();
for (let i = 0; i < n; i++) await dec.run(z, i / n);
console.log(`     inferencia: ${((performance.now() - t0) / n).toFixed(2)} ms/frame (wasm, 1 thread)`);
