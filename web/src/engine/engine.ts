import workletUrl from "./player.worklet.ts?worker&url";
import type { FromWorker, FromWorklet, LoadRequest, SynthParams, ToWorker, ToWorklet } from "./protocol";

export interface EngineEvents {
  onFrame?: (spectrum: Float32Array, inferenceMs: number, synthMs: number) => void;
  onStats?: (stats: FromWorklet) => void;
  onError?: (message: string) => void;
}

/**
 * Motor de audio del hilo principal: AudioContext + AudioWorklet de reproducción + worker de síntesis.
 * El AudioContext corre a la frecuencia de muestreo del modelo (el browser remuestrea a la placa de audio).
 */
export class SynthEngine {
  private readonly worker: Worker;
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private readonly events: EngineEvents;
  private loadId = 0;
  private readonly pendingLoads = new Map<number, { resolve: (ms: number) => void; reject: (e: Error) => void }>();
  private workletConfig: Extract<ToWorklet, { type: "config" }> = { type: "config" };
  private sampleRate = 22050;
  private suspendTimer: number | undefined;

  constructor(events: EngineEvents) {
    this.events = events;
    this.worker = new Worker(new URL("./synth.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<FromWorker>) => this.onWorkerMessage(ev.data);
    this.worker.onerror = (ev) => events.onError?.(ev.message || "Error en el worker de síntesis");
  }

  get audioSampleRate(): number | null {
    return this.ctx?.sampleRate ?? null;
  }

  get outputLatencyMs(): number {
    const ctx = this.ctx;
    if (!ctx) return 0;
    return 1000 * ((ctx.baseLatency || 0) + ((ctx as AudioContext & { outputLatency?: number }).outputLatency || 0));
  }

  private send(msg: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  private onWorkerMessage(msg: FromWorker) {
    switch (msg.type) {
      case "loaded":
        this.pendingLoads.get(msg.requestId)?.resolve(msg.ms);
        this.pendingLoads.delete(msg.requestId);
        break;
      case "loadError":
        this.pendingLoads.get(msg.requestId)?.reject(new Error(msg.message));
        this.pendingLoads.delete(msg.requestId);
        break;
      case "frame":
        this.events.onFrame?.(msg.spectrum, msg.inferenceMs, msg.synthMs);
        break;
      case "error":
        this.events.onError?.(msg.message);
        break;
    }
  }

  /** Carga un par de modelos. Si la frecuencia de muestreo cambia, se recrea el AudioContext. */
  load(request: LoadRequest): Promise<number> {
    const requestId = ++this.loadId;
    this.sampleRate = request.sampleRate;
    if (this.ctx && this.ctx.sampleRate !== request.sampleRate) void this.rebuildAudio();
    const done = new Promise<number>((resolve, reject) => this.pendingLoads.set(requestId, { resolve, reject }));
    this.send({ type: "load", request, requestId });
    return done;
  }

  setParams(params: SynthParams) {
    this.send({ type: "params", params });
  }

  configure(config: Omit<Extract<ToWorklet, { type: "config" }>, "type">) {
    this.workletConfig = { ...this.workletConfig, ...config };
    this.node?.port.postMessage({ type: "config", ...config } satisfies ToWorklet);
  }

  /** Arranca/para el audio. Arrancar tiene que llamarse desde un gesto del usuario (política de autoplay). */
  async setPlaying(playing: boolean) {
    window.clearTimeout(this.suspendTimer);
    if (playing) {
      await this.ensureAudio();
      await this.ctx!.resume();
      this.send({ type: "reset" });
      this.configure({ playing: true });
    } else {
      this.configure({ playing: false });
      // Dejar terminar el fade-out y suspender para no gastar CPU
      this.suspendTimer = window.setTimeout(() => void this.ctx?.suspend(), 300);
    }
  }

  private async ensureAudio() {
    if (this.ctx && this.node) return;
    const ctx = new AudioContext({ sampleRate: this.sampleRate, latencyHint: "interactive" });
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, "asmrs-player", { numberOfInputs: 0, outputChannelCount: [2] });
    node.port.onmessage = (ev: MessageEvent<FromWorklet>) => this.events.onStats?.(ev.data);
    node.connect(ctx.destination);
    const channel = new MessageChannel();
    node.port.postMessage({ type: "connect", port: channel.port1 } satisfies ToWorklet, [channel.port1]);
    this.send({ type: "connect", port: channel.port2 }, [channel.port2]);
    node.port.postMessage(this.workletConfig);
    this.ctx = ctx;
    this.node = node;
  }

  private async rebuildAudio() {
    const wasPlaying = this.workletConfig.playing;
    const ctx = this.ctx;
    this.node?.disconnect();
    this.ctx = null;
    this.node = null;
    await ctx?.close();
    if (wasPlaying) await this.setPlaying(true);
  }
}
