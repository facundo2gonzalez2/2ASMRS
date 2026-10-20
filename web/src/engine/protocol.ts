/** Mensajes entre el hilo principal, el worker de síntesis y el AudioWorklet. */
import type { PhaseMode } from "../dsp/framesynth";

export interface ModelRef {
  id: string;
  weightsUrl: string;
  xMax: number;
  dbMinNorm: number;
}

export interface LoadRequest {
  graphUrl: string;
  a: ModelRef;
  b: ModelRef;
  latentDim: number;
  sampleRate: number;
  winLength: number;
  hopLength: number;
}

export interface SynthParams {
  z: number[];
  alpha: number;
  contrast: number; // escala de xMax (rango en dB del espectro decodificado)
  phaseMode: PhaseMode;
  rtisiLookahead: number;
  rtisiIterations: number;
  glideMs: number; // suavizado de z y α
}

// Main → worker
export type ToWorker =
  | { type: "connect"; port: MessagePort }
  | { type: "load"; request: LoadRequest; requestId: number }
  | { type: "params"; params: SynthParams }
  | { type: "reset" };

// Worker → main
export type FromWorker =
  | { type: "ready" }
  | { type: "loaded"; requestId: number; ms: number }
  | { type: "loadError"; requestId: number; message: string }
  | { type: "frame"; spectrum: Float32Array; inferenceMs: number; synthMs: number }
  | { type: "error"; message: string };

// Worklet ↔ worker (MessageChannel)
export type WorkletToWorker = { type: "need"; n: number };
export type WorkerToWorklet = { type: "block"; data: Float32Array; solicited: boolean };

// Main → worklet (node.port)
export type ToWorklet =
  | { type: "connect"; port: MessagePort }
  | { type: "config"; playing?: boolean; volumeDb?: number; autoGain?: boolean; targetBlocks?: number; hop?: number };

// Worklet → main
export type FromWorklet = { type: "stats"; rmsDb: number; peak: number; queued: number; underruns: number; agcDb: number };
