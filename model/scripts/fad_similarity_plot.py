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
from audio_utils import get_spectrograms_from_audios
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


def _list_valid_audio_files(folder, hps, num_frames):
    folder = Path(folder)
    target_sr = hps["target_sampling_rate"]
    min_samples = num_frames * hps["hop_length"] + hps["win_length"]
    valid = []
    for ext in ("*.wav", "*.mp3"):
        for p in sorted(folder.glob(ext)):
            try:
                info = sf.info(str(p))
            except Exception:
                continue
            n_at_target = int(info.frames * (target_sr / info.samplerate))
            if n_at_target >= min_samples:
                valid.append(p)
    if not valid:
        raise FileNotFoundError(f"Sin audios largos suficientes en {folder} para {num_frames} frames.")
    return valid


def _encode_audio_to_z(model, hps, audio_path, num_frames):
    X, _, _, _ = get_spectrograms_from_audios(
        [Path(audio_path)],
        hps["target_sampling_rate"],
        hps["win_length"],
        hps["hop_length"],
        db_min_norm=hps["db_min_norm"],
        spec_in_db=hps["spec_in_db"],
        normalize_each_audio=hps["normalize_each_audio"],
    )
    if X.shape[0] < num_frames:
        raise ValueError(f"{audio_path}: spec tiene {X.shape[0]} frames, se necesitan {num_frames}")
    X = X[:num_frames]
    with torch.no_grad():
        mu, _ = model.encoder(X)
    return mu


def _compute_z_distribution(model, hps, audio_files):
    sum_z = None
    sum_z_sq = None
    total_frames = 0
    for audio_path in audio_files:
        X, _, _, _ = get_spectrograms_from_audios(
            [Path(audio_path)],
            hps["target_sampling_rate"],
            hps["win_length"],
            hps["hop_length"],
            db_min_norm=hps["db_min_norm"],
            spec_in_db=hps["spec_in_db"],
            normalize_each_audio=hps["normalize_each_audio"],
        )
        with torch.no_grad():
            mu, _ = model.encoder(X)
        if sum_z is None:
            sum_z = torch.zeros(mu.shape[1])
            sum_z_sq = torch.zeros(mu.shape[1])
        sum_z += mu.sum(dim=0).cpu()
        sum_z_sq += (mu**2).sum(dim=0).cpu()
        total_frames += mu.shape[0]
    if total_frames == 0 or sum_z is None or sum_z_sq is None:
        raise ValueError("No se obtuvieron frames para calcular estadísticas de Z.")
    mean = sum_z / total_frames
    var = sum_z_sq / total_frames - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean, std


def _sample_z_from_distribution(mean, std, num_frames):
    latent_dim = mean.shape[0]
    return mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(num_frames, latent_dim)


def main():
    # ── Config ──────────────────────────────────────────
    instrument_b = "voice"
    instrument_a = "piano"
    num_frames = 64
    num_samples = 10
    phase_mode = "pghi"
    interpolation_mode = "slerp"
    z_latent_random = True
    # alphas = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    seed = 0
    ref_path_a = MODEL_DIR / "data_instruments" / instrument_a
    ref_path_b = MODEL_DIR / "data_instruments" / instrument_b  # simetría; no se usa en este script
    # ────────────────────────────────────────────────────
    for source, beta in [
        ("scratch", "no_beta"),
        ("scratch", "beta_0.001"),
        ("checkpoint", "beta_0.001"),
        ("checkpoint", "no_beta"),
    ]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"Cargando modelo {instrument_a}...")
        model_a, hps_a = _load_instrument_model(instrument_a, source, beta)
        print(f"Cargando modelo {instrument_b}...")
        model_b, hps_b = _load_instrument_model(instrument_b, source, beta)

        valid_audios = _list_valid_audio_files(ref_path_a, hps_a, num_frames)
        assert (
            len(valid_audios) >= num_samples
        ), f"Solo {len(valid_audios)} audios válidos en {ref_path_a}, se necesitan {num_samples}"
        rng = np.random.default_rng(seed)
        chosen_audios = [valid_audios[i] for i in rng.choice(len(valid_audios), size=num_samples, replace=False)]
        print(f"Audios elegidos de {ref_path_a.name}: {[p.name for p in chosen_audios]}")

        if z_latent_random:
            stats_audios = valid_audios[: min(25, len(valid_audios))]
            z_mean, z_std = _compute_z_distribution(model_a, hps_a, stats_audios)
            print(f"Z ~ N(μ, σ) ajustada sobre {len(stats_audios)} audios de {instrument_a}")

        assert hps_a["encoder_layers"] == hps_b["encoder_layers"], "Arquitecturas no coinciden"
        assert hps_a["decoder_layers"] == hps_b["decoder_layers"], "Arquitecturas no coinciden"
        assert hps_a["latent_dim"] == hps_b["latent_dim"], "Arquitecturas no coinciden"

        latent_dim = hps_a["latent_dim"]
        xmax_p = hps_a["Xmax"]
        xmax_v = hps_b["Xmax"]

        results = {float(a): {"cos": [], "fad": []} for a in alphas}
        tmpdir = Path(tempfile.mkdtemp(prefix="fad_similarity_plot_"))
        print(f"Directorio temporal: {tmpdir}")

        try:
            for s in range(num_samples):
                print(f"\n── Sample {s + 1}/{num_samples} ──")
                if z_latent_random:
                    z = _sample_z_from_distribution(z_mean, z_std, num_frames)
                else:
                    z = _encode_audio_to_z(model_a, hps_a, chosen_audios[s], num_frames)

                ref_path = _decode_to_wav(model_a, z, xmax_p, hps_a, phase_mode, tmpdir / f"ref_s{s}.wav")
                ref_vec = get_embedding(ref_path)
                ref_mat = get_matrix_embedding(ref_path)

                for a in alphas:
                    a_f = float(a)
                    model_i = interpolar_vae(
                        model_b,
                        model_a,
                        a_f,
                        encoder_layers=hps_a["encoder_layers"],
                        decoder_layers=hps_a["decoder_layers"],
                        latent_dim=latent_dim,
                        interpolation_mode=interpolation_mode,
                    )
                    model_i.eval()
                    xmax_i = (1.0 - a_f) * xmax_v + a_f * xmax_p

                    wav_path = _decode_to_wav(
                        model_i,
                        z,
                        xmax_i,
                        hps_a,
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
            print(f" α {a:.2f}: Cos MERT = {mc:.4f}±{sc:.4f}, Sim FAD = {mf:.4f}±{sf_:.4f}")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = "tab:blue"
        ax1.set_xlabel("α", fontsize=12)
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
            f"Similitud vs α (promedio sobre {num_samples}; {instrument_b}→{instrument_a}, ref={instrument_a}; {source}, {beta})",
            fontsize=13,
        )
        ax1.grid(True, linestyle="--", alpha=0.6)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center left")  # type: ignore

        fig.tight_layout()
        z_tag = "zrandom" if z_latent_random else "zencoded"
        filename = (
            MODEL_DIR
            / f"imgs/fad_similarity2/similarity_vs_fad_{instrument_b}_{instrument_a}_{source}_{beta}_{z_tag}.png"
        )
        plt.savefig(filename)
        print(f"Gráfico guardado como {filename}")


if __name__ == "__main__":
    main()
