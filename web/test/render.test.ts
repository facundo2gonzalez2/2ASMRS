/**
 * Render offline de la cadena completa del worker (decoder ONNX interpolado → magnitud → fase → overlap-add).
 * Verifica que el audio sintetizado tenga el espectro que pidió el decoder y escribe WAVs en test/out/ para
 * escuchar: un Z fijo por modo de fase y un barrido α 0→1 (A puro → B puro).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { FFT } from "../src/dsp/fft";
import { FrameSynth, type PhaseMode, type RtisiConfig } from "../src/dsp/framesynth";
import { InterpolatedDecoder } from "../src/engine/decoder";

const MODELS = new URL("../public/models/", import.meta.url);
const OUT = new URL("./out/", import.meta.url);
mkdirSync(OUT, { recursive: true });
ort.env.wasm.numThreads = 1;

const manifest = JSON.parse(readFileSync(new URL("manifest.json", MODELS), "utf8"));
const model = (id: string) => manifest.models.find((m: { id: string }) => m.id === id);
const bytes = (rel: string) => new Uint8Array(readFileSync(new URL(rel, MODELS)));

const A = model("piano_from_checkpoint_beta_0.001");
const B = model("voice_from_checkpoint_beta_0.001");
const CONTRAST = 1.45;
const { sampleRate: SR, winLength: N, hopLength: H } = A;
const BINS = N / 2 + 1;
const RTISI: RtisiConfig = { lookahead: 2, iterations: 8 };

const dec = await InterpolatedDecoder.create(ort, bytes(manifest.archs[A.arch].graph), bytes(A.weights), bytes(B.weights), A.latentDim);

function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${detail}`);
  if (!ok) process.exitCode = 1;
}

async function render(frames: number, at: (f: number) => { z: number[]; alpha: number }, mode: PhaseMode) {
  const synth = new FrameSynth(N, H);
  const delay = mode === "rtisi" ? RTISI.lookahead : 0;
  const audio = new Float32Array((frames + delay) * H);
  const targets: Float64Array[] = [];
  for (let f = 0; f < frames + delay; f++) {
    const { z, alpha } = at(Math.min(f, frames - 1));
    const y = await dec.run(z, alpha);
    const xMax = ((1 - alpha) * A.xMax + alpha * B.xMax) * CONTRAST;
    const target = new Float64Array(BINS);
    for (let i = 0; i < BINS; i++) target[i] = 10 ** ((Math.min(1, Math.max(0, y[i])) * xMax + A.dbMinNorm) / 20);
    targets.push(target);
    audio.set(synth.process(y, xMax, A.dbMinNorm, mode, RTISI), f * H);
  }
  return { audio: audio.subarray(delay * H), targets };
}

function spectralConvergenceDb(audio: Float32Array, targets: Float64Array[]) {
  const fft = new FFT(N);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  let err = 0;
  let ref = 0;
  for (let f = 4; (f + 1) * H + N <= audio.length; f++) {
    for (let k = 0; k < N; k++) {
      re[k] = audio[f * H + k] * (0.5 - 0.5 * Math.cos((2 * Math.PI * k) / N));
      im[k] = 0;
    }
    fft.transform(re, im);
    const t = targets[f];
    for (let i = 0; i < BINS; i++) {
      err += (Math.hypot(re[i], im[i]) - t[i]) ** 2;
      ref += t[i] ** 2;
    }
  }
  return 10 * Math.log10(err / ref);
}

function writeWav(name: string, audio: Float32Array) {
  let peak = 0;
  for (const v of audio) peak = Math.max(peak, Math.abs(v));
  const scale = peak > 0 ? 0.9 / peak : 1;
  const buf = Buffer.alloc(44 + audio.length * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + audio.length * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(audio.length * 2, 40);
  audio.forEach((v, i) => buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, v * scale)) * 32767), 44 + i * 2));
  writeFileSync(new URL(name, OUT), buf);
}

const seconds = (s: number) => Math.round((s * SR) / H);

// Referencia offline (librosa) sobre este mismo espectro: Griffin-Lim 32 it. ≈ −16.5 dB
const scByMode: Record<string, number> = {};
for (const mode of ["rtisi", "pghi", "random"] as PhaseMode[]) {
  const { audio, targets } = await render(seconds(3), () => ({ z: A.zInit, alpha: 0 }), mode);
  const sc = spectralConvergenceDb(audio, targets);
  const rms = Math.sqrt(audio.reduce((s, v) => s + v * v, 0) / audio.length);
  console.log(`     ${mode}: SC ${sc.toFixed(2)} dB, RMS ${(20 * Math.log10(rms)).toFixed(1)} dBFS`);
  check(`${mode}: audio finito`, Number.isFinite(rms) && rms > 0, `rms ${rms.toExponential(2)}`);
  scByMode[mode] = sc;
  writeWav(`piano_zinit_${mode}.wav`, audio);
}

check("rtisi reproduce el espectro del decoder", scByMode.rtisi < -12, `${scByMode.rtisi.toFixed(2)} dB`);
check("pghi mejor que fase aleatoria", scByMode.pghi < scByMode.random - 3, `${scByMode.pghi.toFixed(2)} dB`);

// Barrido: 1 s de A, 4 s de transición (α y Z de zInit A a zInit B), 1 s de B
const hold = seconds(1);
const ramp = seconds(4);
const sweep = await render(2 * hold + ramp, (f) => {
  const alpha = Math.min(1, Math.max(0, (f - hold) / ramp));
  return { alpha, z: A.zInit.map((v: number, i: number) => (1 - alpha) * v + alpha * B.zInit[i]) };
}, "rtisi");
writeWav("sweep_piano_to_voice_rtisi.wav", sweep.audio);
console.log(`     WAVs en ${OUT.pathname}`);
