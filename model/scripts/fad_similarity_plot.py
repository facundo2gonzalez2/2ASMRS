import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.spatial.distance import cosine

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from VariationalAutoEncoder import VariationalAutoEncoder
from audio_comparator import (
    get_audio_similarity_fad,
    get_embedding,
    get_matrix_embedding,
)
from experiments.interpolate import interpolar_vae
from scripts.vae_predict import predict_audio


def _load_instrument_model(instrument, source, beta):
    model_dir = (
        MODEL_DIR / f"inference_models/instruments_from_{source}" / f"{instrument}_from_{source}_{beta}" / "version_0"
    )
    checkpoint_path = list((model_dir / "checkpoints").glob("*.ckpt"))[0]
    with open(model_dir / "hparams.yaml") as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)

    model = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    model.decoder.eval()
    return model, hps


def _decode_to_wav(model, z, xmax, hps, phase_option, out_path):
    with torch.no_grad():
        raw = model.decoder(z)
        spec = raw * xmax
        spec = torch.clamp(
            torch.nan_to_num(spec, nan=0.0, posinf=xmax, neginf=0.0),
            min=0.0,
            max=xmax,
        )
    audio = predict_audio(
        predicted_specgram=spec,
        hps=hps,
        phase_option=phase_option,
        frames=z.shape[0],
        return_audio=True,
    )
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
    sf.write(str(out_path), audio, hps["target_sampling_rate"])
    return str(out_path)


def main():
    # ── Config ──────────────────────────────────────────
    source = "checkpoint"
    beta = "beta_0.001"
    num_frames = 64
    num_samples = 10
    phase_mode = "pghi"
    interpolation_mode = "slerp"
    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2)
    seed = 0
    # ────────────────────────────────────────────────────

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Cargando modelo piano...")
    piano_model, hps_p = _load_instrument_model("piano", source, beta)
    print("Cargando modelo voice...")
    voice_model, hps_v = _load_instrument_model("voice", source, beta)

    assert hps_p["encoder_layers"] == hps_v["encoder_layers"], "Arquitecturas no coinciden"
    assert hps_p["decoder_layers"] == hps_v["decoder_layers"], "Arquitecturas no coinciden"
    assert hps_p["latent_dim"] == hps_v["latent_dim"], "Arquitecturas no coinciden"

    latent_dim = hps_p["latent_dim"]
    xmax_p = hps_p["Xmax"]
    xmax_v = hps_v["Xmax"]

    results = {float(a): {"cos": [], "fad": []} for a in alphas}
    tmpdir = Path(tempfile.mkdtemp(prefix="fad_similarity_plot_"))
    print(f"Directorio temporal: {tmpdir}")

    try:
        for s in range(num_samples):
            print(f"\n── Sample {s + 1}/{num_samples} ──")
            z = torch.randn(num_frames, latent_dim)

            ref_path = _decode_to_wav(piano_model, z, xmax_p, hps_p, phase_mode, tmpdir / f"ref_s{s}.wav")
            ref_vec = get_embedding(ref_path)
            ref_mat = get_matrix_embedding(ref_path)

            for a in alphas:
                a_f = float(a)
                model_i = interpolar_vae(
                    voice_model,
                    piano_model,
                    a_f,
                    encoder_layers=hps_p["encoder_layers"],
                    decoder_layers=hps_p["decoder_layers"],
                    latent_dim=latent_dim,
                    interpolation_mode=interpolation_mode,
                )
                model_i.eval()
                xmax_i = (1.0 - a_f) * xmax_v + a_f * xmax_p

                wav_path = _decode_to_wav(
                    model_i,
                    z,
                    xmax_i,
                    hps_p,
                    phase_mode,
                    tmpdir / f"s{s}_a{a_f:.2f}.wav",
                )
                cos_sim = 1.0 - float(cosine(ref_vec, get_embedding(wav_path)))
                fad_sim = float(get_audio_similarity_fad(ref_mat, get_matrix_embedding(wav_path)))
                results[a_f]["cos"].append(cos_sim)
                results[a_f]["fad"].append(fad_sim)
                print(f"  alpha={a_f:.2f}  cos={cos_sim:.4f}  fad_sim={fad_sim:.4f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    alpha_list = [float(a) for a in alphas]
    mean_cos = np.array([np.mean(results[a]["cos"]) for a in alpha_list])
    std_cos = np.array([np.std(results[a]["cos"]) for a in alpha_list])
    mean_fad = np.array([np.mean(results[a]["fad"]) for a in alpha_list])
    std_fad = np.array([np.std(results[a]["fad"]) for a in alpha_list])

    print("\n── Resultados promedio ──")
    for a, mc, sc, mf, sf_ in zip(alpha_list, mean_cos, std_cos, mean_fad, std_fad):
        print(f" Alpha {a:.2f}: Cos MERT = {mc:.4f}±{sc:.4f}, Sim FAD = {mf:.4f}±{sf_:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Alfa", fontsize=12)
    ax1.set_ylabel("Similitud de Coseno (MERT)", color=color1, fontsize=12)
    (line1,) = ax1.plot(
        alpha_list,
        mean_cos,
        marker="o",
        color=color1,
        linewidth=2,
        label="Similitud MERT",
    )
    ax1.fill_between(alpha_list, mean_cos - std_cos, mean_cos + std_cos, color=color1, alpha=0.15)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(alpha_list)

    for x, y in zip(alpha_list, mean_cos):
        ax1.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=color1,
            fontsize=9,
            fontweight="bold",
        )

    ax2 = ax1.twinx()

    color2 = "tab:red"
    ax2.set_ylabel("Similitud FAD", color=color2, fontsize=12)
    (line2,) = ax2.plot(
        alpha_list,
        mean_fad,
        marker="s",
        color=color2,
        linewidth=2,
        linestyle="--",
        label="Similitud FAD",
    )
    ax2.fill_between(alpha_list, mean_fad - std_fad, mean_fad + std_fad, color=color2, alpha=0.15)
    ax2.tick_params(axis="y", labelcolor=color2)

    for x, y in zip(alpha_list, mean_fad):
        ax2.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color=color2,
            fontsize=9,
            fontweight="bold",
        )

    plt.title(
        f"Similitud vs Alfa (promedio sobre {num_samples}; voz→piano, ref=piano; {source}, {beta})",
        fontsize=13,
    )
    ax1.grid(True, linestyle="--", alpha=0.6)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")  # type: ignore

    fig.tight_layout()
    filename = MODEL_DIR / f"imgs/similarity_vs_fad_{source}_{beta}.png"
    plt.savefig(filename)
    print(f"Gráfico guardado como {filename}")


if __name__ == "__main__":
    main()
