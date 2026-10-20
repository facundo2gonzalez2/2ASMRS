import type * as OrtModule from "onnxruntime-web";

type Ort = typeof OrtModule;

/**
 * Decoder interpolado (grafo de model/web_export.py) sobre ONNX Runtime Web.
 *
 * El grafo es único por arquitectura y referencia los pesos de los dos checkpoints como datos externos "A.bin" y
 * "B.bin"; acá se les asignan los `.bin` de los checkpoints elegidos. Los ángulos SLERP solo dependen de los pesos,
 * así que ORT los pliega como constantes al crear la sesión.
 */
export class InterpolatedDecoder {
  private readonly ort: Ort;
  private readonly session: OrtModule.InferenceSession;
  readonly latentDim: number;
  private readonly zData: Float32Array;
  private readonly zTensor: OrtModule.Tensor;
  private readonly alphaData = new Float32Array(1);
  private readonly alphaTensor: OrtModule.Tensor;

  private constructor(ort: Ort, session: OrtModule.InferenceSession, latentDim: number) {
    this.ort = ort;
    this.session = session;
    this.latentDim = latentDim;
    this.zData = new Float32Array(latentDim);
    this.zTensor = new ort.Tensor("float32", this.zData, [1, latentDim]);
    this.alphaTensor = new ort.Tensor("float32", this.alphaData, [1]);
  }

  static async create(
    ort: Ort,
    graph: Uint8Array,
    weightsA: Uint8Array,
    weightsB: Uint8Array,
    latentDim: number,
  ): Promise<InterpolatedDecoder> {
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      externalData: [
        { path: "A.bin", data: weightsA },
        { path: "B.bin", data: weightsB },
      ],
    });
    return new InterpolatedDecoder(ort, session, latentDim);
  }

  /** Un frame: z (latentDim) y α → espectro normalizado (winLength/2+1). */
  async run(z: ArrayLike<number>, alpha: number): Promise<Float32Array> {
    for (let i = 0; i < this.latentDim; i++) this.zData[i] = z[i] ?? 0;
    this.alphaData[0] = alpha;
    const out = await this.session.run({ z: this.zTensor, alpha: this.alphaTensor });
    return out.spectrum.data as Float32Array;
  }

  /** Varios frames de una vez (z aplanado frames×latentDim). */
  async runBatch(z: Float32Array, alpha: number): Promise<Float32Array> {
    const frames = z.length / this.latentDim;
    const zt = new this.ort.Tensor("float32", z, [frames, this.latentDim]);
    this.alphaData[0] = alpha;
    const out = await this.session.run({ z: zt, alpha: this.alphaTensor });
    return out.spectrum.data as Float32Array;
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}
