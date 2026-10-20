/// <reference lib="webworker" />
/**
 * Worker de síntesis: decoder interpolado (ONNX Runtime Web) + reconstrucción de fase + overlap-add.
 *
 * El AudioWorklet pide bloques de `hopLength` muestras; por cada pedido se genera un frame con los últimos
 * parámetros recibidos. Al cambiar de modelos, antes de crear la sesión nueva (que bloquea este hilo unos cientos
 * de ms) se mandan bloques extra con la sesión vieja para que el audio no se corte.
 */
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { FrameSynth } from "../dsp/framesynth";
import { InterpolatedDecoder } from "./decoder";
import type { FromWorker, LoadRequest, SynthParams, ToWorker, WorkerToWorklet, WorkletToWorker } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

ort.env.wasm.numThreads = 1; // multi-thread pide cross-origin isolation y no hace falta para batch 1
ort.env.wasm.wasmPaths = { wasm: wasmUrl };

const WEIGHTS_CACHE_SIZE = 6;

let port: MessagePort | null = null;
let decoder: InterpolatedDecoder | null = null;
let current: LoadRequest | null = null;
let synth: FrameSynth | null = null;
let params: SynthParams = {
  z: [0, 0, 0, 0],
  alpha: 0,
  contrast: 1,
  phaseMode: "rtisi",
  rtisiLookahead: 2,
  rtisiIterations: 8,
  glideMs: 40,
};
const smoothed = { z: [0, 0, 0, 0], alpha: 0, init: false };
let pendingBlocks = 0;
let pumping = false;
let lastLoadMs = 600;
let latestLoadId = -1;

// Toda la generación y el cambio de sesión pasan por esta cola: una sola corrida de ORT a la vez y los frames
// del overlap-add en orden.
let chain: Promise<unknown> = Promise.resolve();
function exclusive<T>(fn: () => Promise<T>): Promise<T> {
  const result = chain.then(fn);
  chain = result.catch(() => undefined);
  return result;
}

const weightsCache = new Map<string, Promise<Uint8Array>>();
const graphCache = new Map<string, Promise<Uint8Array>>();

function post(msg: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

function cached(cache: Map<string, Promise<Uint8Array>>, url: string, limit = Infinity): Promise<Uint8Array> {
  let entry = cache.get(url);
  if (entry) {
    cache.delete(url); // LRU: reinsertar al final
  } else {
    entry = fetchBytes(url);
    entry.catch(() => cache.delete(url));
  }
  cache.set(url, entry);
  while (cache.size > limit) cache.delete(cache.keys().next().value!);
  return entry;
}

async function load(request: LoadRequest, requestId: number) {
  latestLoadId = requestId;
  try {
    const [graph, wa, wb] = await Promise.all([
      cached(graphCache, request.graphUrl),
      cached(weightsCache, request.a.weightsUrl, WEIGHTS_CACHE_SIZE),
      cached(weightsCache, request.b.weightsUrl, WEIGHTS_CACHE_SIZE),
    ]);
    if (requestId !== latestLoadId) return; // llegó un pedido más nuevo mientras se descargaba
    await exclusive(async () => {
      if (requestId !== latestLoadId) return;
      // Colchón de audio con la sesión actual mientras se crea la nueva (bloquea este hilo)
      if (decoder && current && port) {
        const hopMs = (1000 * current.hopLength) / current.sampleRate;
        await generate(Math.min(80, Math.ceil((lastLoadMs * 1.5) / hopMs) + 4), false);
      }
      const t0 = performance.now();
      const next = await InterpolatedDecoder.create(ort, graph, wa, wb, request.latentDim);
      lastLoadMs = performance.now() - t0;
      const old = decoder;
      decoder = next;
      const sameFrames = current && current.winLength === request.winLength && current.hopLength === request.hopLength;
      current = request;
      if (!sameFrames || !synth) synth = new FrameSynth(request.winLength, request.hopLength);
      await old?.release();
    });
    post({ type: "loaded", requestId, ms: lastLoadMs });
  } catch (e) {
    post({ type: "loadError", requestId, message: e instanceof Error ? e.message : String(e) });
  }
}

/** Genera `n` bloques y los manda al worklet. */
async function generate(n: number, solicited: boolean) {
  for (let i = 0; i < n; i++) {
    const block = await nextBlock();
    const msg: WorkerToWorklet = { type: "block", data: block, solicited };
    port?.postMessage(msg, [block.buffer]);
  }
}

async function nextBlock(): Promise<Float32Array> {
  if (!decoder || !current || !synth) return new Float32Array(current?.hopLength ?? 512);

  // Suavizado exponencial de z y α (un paso por frame)
  const hopSec = current.hopLength / current.sampleRate;
  const k = params.glideMs > 0 ? 1 - Math.exp(-hopSec / (params.glideMs / 1000)) : 1;
  const dim = current.latentDim;
  if (!smoothed.init || smoothed.z.length !== dim) {
    smoothed.z = Array.from({ length: dim }, (_, i) => params.z[i] ?? 0);
    smoothed.alpha = params.alpha;
    smoothed.init = true;
  }
  for (let i = 0; i < dim; i++) smoothed.z[i] += k * ((params.z[i] ?? 0) - smoothed.z[i]);
  smoothed.alpha += k * (params.alpha - smoothed.alpha);
  const alpha = smoothed.alpha;

  const t0 = performance.now();
  const y = await decoder.run(smoothed.z, alpha);
  const inferenceMs = performance.now() - t0;
  const t1 = performance.now();

  // Como en app.py: xMax de entrenamiento interpolado entre A y B, escalado por el contraste
  const xMax = ((1 - alpha) * current.a.xMax + alpha * current.b.xMax) * params.contrast;
  const dbMin = (1 - alpha) * current.a.dbMinNorm + alpha * current.b.dbMinNorm;
  const out = synth
    .process(y, xMax, dbMin, params.phaseMode, { lookahead: params.rtisiLookahead, iterations: params.rtisiIterations })
    .slice();

  const synthMs = performance.now() - t1;

  const spectrum = y.slice();
  post({ type: "frame", spectrum, inferenceMs, synthMs }, [spectrum.buffer]);
  return out;
}

async function pump() {
  if (pumping) return;
  pumping = true;
  try {
    while (pendingBlocks > 0) {
      pendingBlocks--;
      await exclusive(() => generate(1, true));
    }
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
  } finally {
    pumping = false;
  }
}

self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data;
  switch (msg.type) {
    case "connect":
      port = msg.port;
      port.onmessage = (e: MessageEvent<WorkletToWorker>) => {
        if (e.data.type === "need") {
          pendingBlocks += e.data.n;
          void pump();
        }
      };
      break;
    case "load":
      void load(msg.request, msg.requestId);
      break;
    case "params":
      params = msg.params;
      break;
    case "reset":
      void exclusive(async () => {
        synth?.reset();
        smoothed.init = false;
      });
      break;
  }
};

post({ type: "ready" });
