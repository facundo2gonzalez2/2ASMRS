/**
 * Test de la reconstrucción de fase: se sintetiza audio a partir de las magnitudes STFT de un audio real
 * (fixtures/stft_target.f32, generado por make_fixtures.py) y se mide la consistencia espectral
 * SC = ‖ |STFT(salida)| − |X| ‖ / ‖X‖ en dB. Más negativo = mejor. RTPGHI tiene que quedar bien por debajo de la
 * fase aleatoria (que es lo que hacía el sintetizador C++) y RTISI-LA por debajo de RTPGHI.
 * Referencia offline (librosa, mismo audio): Griffin-Lim 32 it. ≈ −23 dB, 100 it. ≈ −32 dB.
 */
import { readFileSync } from "node:fs";
import { FFT } from "../src/dsp/fft";
import { FrameSynth, type PhaseMode, type RtisiConfig } from "../src/dsp/framesynth";

const N = 2048;
const H = 512;
const BINS = N / 2 + 1;

const raw = readFileSync(new URL("./fixtures/stft_target.f32", import.meta.url));
const target = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const frames = target.length / BINS;

const RTISI: RtisiConfig = { lookahead: 2, iterations: 8 };

/** Audio sintetizado, alineado con el target (se descarta el retardo del lookahead de RTISI). */
function synthesize(mode: PhaseMode): Float32Array {
  const synth = new FrameSynth(N, H);
  const delay = mode === "rtisi" ? RTISI.lookahead : 0;
  const out = new Float32Array((frames + delay) * H);
  const logMag = new Float64Array(BINS);
  for (let f = 0; f < frames + delay; f++) {
    const src = Math.min(f, frames - 1);
    for (let i = 0; i < BINS; i++) logMag[i] = Math.log(target[src * BINS + i] + 1e-9);
    out.set(synth.processLogMagnitude(logMag, mode, RTISI), f * H);
  }
  return out.subarray(delay * H);
}

function spectralConvergenceDb(audio: Float32Array): number {
  const fft = new FFT(N);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  let err = 0;
  let ref = 0;
  // Ignorar los primeros/últimos frames (overlap-add incompleto)
  for (let f = 4; f + 4 < frames && (f + 1) * H + N <= audio.length; f++) {
    for (let k = 0; k < N; k++) {
      re[k] = audio[f * H + k] * (0.5 - 0.5 * Math.cos((2 * Math.PI * k) / N));
      im[k] = 0;
    }
    fft.transform(re, im);
    for (let i = 0; i < BINS; i++) {
      const mag = Math.hypot(re[i], im[i]);
      const t = target[f * BINS + i];
      err += (mag - t) ** 2;
      ref += t * t;
    }
  }
  return 10 * Math.log10(err / ref);
}

function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${detail}`);
  if (!ok) process.exitCode = 1;
}

// FFT ida y vuelta
{
  const fft = new FFT(N);
  const re = Float64Array.from({ length: N }, () => Math.random() - 0.5);
  const im = new Float64Array(N);
  const orig = re.slice();
  fft.transform(re, im);
  fft.transform(re, im, true);
  const maxErr = Math.max(...orig.map((v, i) => Math.abs(v - re[i])));
  check("fft roundtrip", maxErr < 1e-12, `max err ${maxErr.toExponential(2)}`);
}

const results: Record<string, number> = {};
for (const mode of ["random", "pghi", "rtisi"] as PhaseMode[]) {
  const t0 = performance.now();
  const audio = synthesize(mode);
  const ms = (performance.now() - t0) / frames;
  results[mode] = spectralConvergenceDb(audio);
  console.log(`     ${mode}: SC ${results[mode].toFixed(2)} dB, ${ms.toFixed(3)} ms/frame`);
}
check("pghi mejor que fase aleatoria", results.pghi < results.random - 6, `${results.pghi.toFixed(2)} vs ${results.random.toFixed(2)} dB`);
check("pghi consistente", results.pghi < -15, `${results.pghi.toFixed(2)} dB`);
check("rtisi mejor que pghi", results.rtisi < results.pghi - 3, `${results.rtisi.toFixed(2)} vs ${results.pghi.toFixed(2)} dB`);
