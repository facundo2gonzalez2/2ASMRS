import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from audio_comparator import get_audio_similarity_fad, get_cosine_similarity, get_matrix_embedding  # noqa: E402
from experiments.interpolate import interpolar_vae  # noqa: E402
from scripts.fad_similarity_plot import (  # noqa: E402
    _compute_z_distribution,
    _decode_to_wav,
    _encode_audio_to_z,
    _list_valid_audio_files,
    _load_instrument_model,
    _sample_z_from_distribution,
)

CONFIG_STYLES = {
    ("scratch", "no_beta"): dict(color="tab:blue", marker="o", label="scratch, no β"),
    ("scratch", "beta_0.001"): dict(color="tab:orange", marker="s", label="scratch, β=0.001"),
    ("checkpoint", "no_beta"): dict(color="tab:green", marker="^", label="checkpoint, no β"),
    ("checkpoint", "beta_0.001"): dict(color="tab:red", marker="D", label="checkpoint, β=0.001"),
}

INSTRUMENT_STYLES = {
    "voice": dict(color="tab:purple", marker="o", label="voice"),
    "guitar": dict(color="tab:olive", marker="s", label="guitar"),
    "bass": dict(color="tab:cyan", marker="^", label="bass"),
}


def main():
    # ── Config ──────────────────────────────────────────
    z_latent_random = False
    similarity_mode = "fad"  # "fad" o "cos"
    instrument_goal = "piano"
    source_instruments = ["voice", "guitar", "bass"]
    num_frames = 64
    num_samples = 10
    phase_mode = "pghi"
    interpolation_mode = "slerp"
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    seed = 0
    per_instrument_source_training = "checkpoint"  # "scratch" | "checkpoint"
    per_instrument_beta = "beta_0.001"  # "no_beta" | "beta_0.001"
    ref_path_goal = MODEL_DIR / "data_instruments" / instrument_goal
    assert similarity_mode in ("fad", "cos"), f"similarity_mode inválido: {similarity_mode}"
    assert (
        per_instrument_source_training,
        per_instrument_beta,
    ) in CONFIG_STYLES, "Config per-instrumento inválida"
    # ────────────────────────────────────────────────────
    alpha_list = [float(a) for a in alphas]
    sim_results = {}  # (source_training, beta) -> (mean_sim, std_sim)
    per_instrument_results = None  # inst -> (mean_per_alpha, std_per_alpha) para la config fija

    for source_training, beta in CONFIG_STYLES.keys():
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n══════ Config: {source_training}, {beta} ══════")
        print(f"Cargando modelo goal ({instrument_goal})...")
        model_goal, hps_goal = _load_instrument_model(instrument_goal, source_training, beta)

        source_models = {}
        for inst in source_instruments:
            print(f"Cargando modelo source ({inst})...")
            m, hps = _load_instrument_model(inst, source_training, beta)
            assert hps["encoder_layers"] == hps_goal["encoder_layers"], "Arquitecturas no coinciden"
            assert hps["decoder_layers"] == hps_goal["decoder_layers"], "Arquitecturas no coinciden"
            assert hps["latent_dim"] == hps_goal["latent_dim"], "Arquitecturas no coinciden"
            source_models[inst] = (m, hps)

        latent_dim = hps_goal["latent_dim"]
        xmax_goal = hps_goal["Xmax"]

        valid = _list_valid_audio_files(ref_path_goal, hps_goal, num_frames)
        assert (
            len(valid) >= num_samples
        ), f"Solo {len(valid)} audios válidos en {ref_path_goal}, se necesitan {num_samples}"
        rng = np.random.default_rng(seed)
        chosen_audios = [valid[i] for i in rng.choice(len(valid), size=num_samples, replace=False)]
        print(f"Audios elegidos de {ref_path_goal.name}: {[p.name for p in chosen_audios]}")

        if z_latent_random:
            stats_audios = valid[: min(25, len(valid))]
            z_mean, z_std = _compute_z_distribution(model_goal, hps_goal, stats_audios)
            print(f"Z ~ N(μ, σ) ajustada sobre {len(stats_audios)} audios de {instrument_goal}")

        results = {inst: {float(a): [] for a in alphas} for inst in source_instruments}
        tmpdir = Path(tempfile.mkdtemp(prefix="fad_similarity_agg_"))
        print(f"Directorio temporal: {tmpdir}")

        try:
            print("\n── Pre-computando Z y referencias del goal ──")
            zs, ref_wavs, ref_mats = [], [], []
            for s in range(num_samples):
                if z_latent_random:
                    print(f"  ref sample {s + 1}/{num_samples}: (Z ~ N(μ, σ) de {instrument_goal})")
                    z = _sample_z_from_distribution(z_mean, z_std, num_frames)
                else:
                    print(f"  ref sample {s + 1}/{num_samples}: {chosen_audios[s].name}")
                    z = _encode_audio_to_z(model_goal, hps_goal, chosen_audios[s], num_frames)

                ref_wav = _decode_to_wav(model_goal, z, xmax_goal, hps_goal, phase_mode, tmpdir / f"ref_s{s}.wav")
                zs.append(z)
                ref_wavs.append(ref_wav)
                if similarity_mode == "fad":
                    ref_mats.append(get_matrix_embedding(ref_wav))

            for source_inst in source_instruments:
                model_b, hps_b = source_models[source_inst]
                xmax_b = hps_b["Xmax"]
                print(f"\n══ Source: {source_inst} → {instrument_goal} ══")
                for s in range(num_samples):
                    print(f"\n── Sample {s + 1}/{num_samples} ──")
                    z = zs[s]
                    for a in alphas:
                        a_f = float(a)
                        model_i = interpolar_vae(
                            model_b,
                            model_goal,
                            a_f,
                            encoder_layers=hps_goal["encoder_layers"],
                            decoder_layers=hps_goal["decoder_layers"],
                            latent_dim=latent_dim,
                            interpolation_mode=interpolation_mode,
                        )
                        model_i.eval()
                        xmax_i = (1.0 - a_f) * xmax_b + a_f * xmax_goal

                        wav_path = _decode_to_wav(
                            model_i,
                            z,
                            xmax_i,
                            hps_goal,
                            phase_mode,
                            tmpdir / f"{source_inst}_s{s}_a{a_f:.2f}.wav",
                        )
                        if similarity_mode == "fad":
                            sim = float(get_audio_similarity_fad(ref_mats[s], get_matrix_embedding(wav_path)))
                        else:
                            sim = float(get_cosine_similarity(ref_wavs[s], wav_path))
                        results[source_inst][a_f].append(sim)
                        print(f"  alpha={a_f:.2f}  {similarity_mode}_sim={sim:.4f}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        per_inst_sim = np.array([[np.mean(results[inst][a]) for a in alpha_list] for inst in source_instruments])
        mean_sim = per_inst_sim.mean(axis=0)
        std_sim = per_inst_sim.std(axis=0)

        sim_label = "FAD" if similarity_mode == "fad" else "Cos MERT"
        print(f"\n── Medias {sim_label} por instrumento ──")
        for i, inst in enumerate(source_instruments):
            sim_str = ", ".join(f"{v:.3f}" for v in per_inst_sim[i])
            print(f"  {inst:6s}  {similarity_mode}=[{sim_str}]")

        print(f"\n── {sim_label} promedio entre instrumentos ──")
        for a, mf, sf_ in zip(alpha_list, mean_sim, std_sim):
            print(f" α {a:.2f}: Sim {sim_label} = {mf:.4f}±{sf_:.4f}")

        sim_results[(source_training, beta)] = (mean_sim, std_sim)

        if (source_training, beta) == (per_instrument_source_training, per_instrument_beta):
            per_instrument_results = {
                inst: (
                    np.array([np.mean(results[inst][a]) for a in alpha_list]),
                    np.array([np.std(results[inst][a]) for a in alpha_list]),
                )
                for inst in source_instruments
            }

    # ... (código anterior) ...

    fig, ax = plt.subplots(figsize=(11, 6))

    for cfg, (mean_sim, std_sim) in sim_results.items():
        style = CONFIG_STYLES[cfg]

        # Extraemos el color de estilo para aplicarlo también al sombreado
        color = style.get("color")

        # 1. Graficamos solo la línea principal (sin errorbar)
        ax.plot(alpha_list, mean_sim, linewidth=2, **style)  # type: ignore

        # 2. Usamos fill_between para el área de desviación estándar
        ax.fill_between(
            alpha_list,
            mean_sim - std_sim,
            mean_sim + std_sim,
            color=color,
            alpha=0.15,  # Transparencia alta para que se vean las áreas superpuestas
            edgecolor="none",
        )

    sim_label = "FAD" if similarity_mode == "fad" else "Cos MERT"
    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel(f"Similitud {sim_label}", fontsize=12)
    ax.set_xticks(alpha_list)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    plt.title(
        f"Similitud {sim_label} vs α hacia {instrument_goal} "
        f"(promedio sobre {len(source_instruments)} sources × {num_samples} samples)",
        fontsize=12,
    )
    fig.tight_layout()
    # ... (código de guardado) ...

    out_dir = MODEL_DIR / "imgs/fad_similarity_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    z_tag = "zrandom" if z_latent_random else "zencoded"
    sim_tag = "fad" if similarity_mode == "fad" else "cos"
    filename = out_dir / f"similarity_vs_{sim_tag}_aggregated_to_{instrument_goal}_{z_tag}.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"\nGráfico guardado como {filename}")

    assert per_instrument_results is not None, "per_instrument_results no fue capturado"
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for inst in source_instruments:
        mean_i, std_i = per_instrument_results[inst]
        style = INSTRUMENT_STYLES[inst]
        color = style.get("color")
        ax2.plot(alpha_list, mean_i, linewidth=2, **style)  # type: ignore
        ax2.fill_between(
            alpha_list,
            mean_i - std_i,
            mean_i + std_i,
            color=color,
            alpha=0.15,
            edgecolor="none",
        )

    ax2.set_xlabel("α", fontsize=12)
    ax2.set_ylabel(f"Similitud {sim_label}", fontsize=12)
    ax2.set_xticks(alpha_list)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", fontsize=10)
    plt.title(
        f"Similitud {sim_label} por instrumento vs α hacia {instrument_goal} "
        f"({per_instrument_source_training}, {per_instrument_beta}, "
        f"promedio sobre {num_samples} samples)",
        fontsize=12,
    )
    fig2.tight_layout()

    filename2 = out_dir / (
        f"similarity_vs_{sim_tag}_per_instrument_to_{instrument_goal}"
        f"_{per_instrument_source_training}_{per_instrument_beta}_{z_tag}.png"
    )
    plt.savefig(filename2)
    plt.close(fig2)
    print(f"Gráfico per-instrumento guardado como {filename2}")


if __name__ == "__main__":
    main()
