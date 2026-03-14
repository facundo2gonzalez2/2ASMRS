import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

from experiments.interpolate_small import interpolar_vae
from scripts.vae_predict import predict_audio
from VariationalAutoEncoder import VariationalAutoEncoder

# --- Constantes ---

INSTRUMENTS = ["piano", "guitar", "vocals", "bass"]
INSTRUMENT_DIR_NAMES = {
    "piano": "piano",
    "guitar": "guitar",
    "vocals": "voice",
    "bass": "bass",
}
XMAX_DEFAULTS = {
    "piano": 150.0,
    "guitar": 150.0,
    "vocals": 140.0,
    "bass": 120.0,
}
INITIAL_Z = [0.0, 0.0, 0.0, 0.0]
Z_INIT_BY_INSTRUMENT = {
    "piano": [-0.00056, 0.10692, -0.02252, -0.09278],
    "guitar": [-0.00103, 0.1799, -0.11215, -0.21214],
    "vocals": [0.00184, 0.05723, 0.19695, 0.62355],
    "bass": [7e-05, 0.14084, -2e-05, 0.06244],
}


NUM_FRAMES = 64


# --- Carga y cache de modelos ---


@st.cache_resource
def load_model(instrument_key: str, beta_config: str, training_source: str):
    """Carga un modelo VAE desde checkpoint. Resultado cacheado por Streamlit."""
    dir_name = INSTRUMENT_DIR_NAMES[instrument_key]
    model_dir = Path(
        "model",
        "inference_models",
        f"instruments_from_{training_source}",
        f"{dir_name}_from_{training_source}_{beta_config}",
        "version_0",
    )

    if not model_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del modelo: {model_dir}")

    ckpt_files = list(Path(model_dir, "checkpoints").glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en: {model_dir / 'checkpoints'}"
        )
    checkpoint_path = ckpt_files[0]

    hparams_path = model_dir / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"No se encontró hparams.yaml en: {model_dir}")

    with open(hparams_path) as f:
        hps = yaml.load(f, Loader=yaml.FullLoader)

    model = VariationalAutoEncoder(
        encoder_layers=hps["encoder_layers"],
        decoder_layers=hps["decoder_layers"],
        latent_dim=hps["latent_dim"],
        checkpoint_path=checkpoint_path,
    )
    model.eval()

    return model, hps


@st.cache_resource
def get_interpolated_model(
    inst_a_key: str,
    inst_b_key: str,
    beta_config: str,
    training_source: str,
    alpha: float,
):
    """Interpola dos modelos VAE con SLERP. Resultado cacheado."""
    model_a, hps_a = load_model(inst_a_key, beta_config, training_source)
    model_b, hps_b = load_model(inst_b_key, beta_config, training_source)

    # Si es el mismo instrumento, devolver modelo A directamente
    if inst_a_key == inst_b_key:
        return model_a, hps_a

    modelo = interpolar_vae(
        model_a,
        model_b,
        alpha,
        encoder_layers=hps_a["encoder_layers"],
        decoder_layers=hps_a["decoder_layers"],
        latent_dim=hps_a["latent_dim"],
        interpolation_mode="slerp",
    )
    modelo.eval()

    return modelo, hps_a


def generate_audio(model, hps, z_values, xmax, num_frames):
    """Genera audio a partir del decoder del modelo interpolado."""
    model.eval()
    model.decoder.eval()

    z = torch.tensor([z_values], dtype=torch.float32)
    z = z.repeat(num_frames, 1)

    with torch.no_grad():
        predicted_specgram = model.decoder(z) * xmax
        predicted_specgram = torch.nan_to_num(
            predicted_specgram, nan=0.0, posinf=xmax, neginf=0.0
        )
        predicted_specgram = torch.clamp(predicted_specgram, min=0.0, max=xmax)

    phase_option = "griffinlim"
    try:
        audio = predict_audio(
            predicted_specgram=predicted_specgram,
            hps=hps,
            phase_option=phase_option,
            frames=num_frames,
            return_audio=True,
        )
    except Exception:
        audio = predict_audio(
            predicted_specgram=predicted_specgram,
            hps=hps,
            phase_option="random",
            frames=num_frames,
            return_audio=True,
        )

    if audio is None:
        return np.zeros(1000, dtype=np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


# --- UI ---

st.set_page_config(page_title="VAE Audio Interpolator", layout="wide")
st.title("VAE Audio Interpolator")

play_enabled = st.toggle("Play", value=st.session_state.get("play_enabled", False))
st.session_state["play_enabled"] = play_enabled

# Sidebar: controles
with st.sidebar:
    st.header("Instrumentos")
    inst_a = st.selectbox("Instrumento A", INSTRUMENTS, index=0)
    inst_b = st.selectbox("Instrumento B", INSTRUMENTS, index=1)

    st.header("Configuración de entrenamiento")
    use_beta = st.toggle("Beta encoding (beta_0.001)", value=True)
    use_checkpoint = st.toggle("Desde checkpoint", value=True)

    beta_config = "beta_0.001" if use_beta else "no_beta"
    training_source = "checkpoint" if use_checkpoint else "scratch"

    st.header("Interpolación")
    if inst_a == inst_b:
        st.info("Ambos instrumentos son iguales. Alpha no tendrá efecto.")
    alpha = st.slider("Alpha", 0.0, 1.0, 0.5, step=0.01)

    st.header("Espacio latente (Z)")
    z_defaults = Z_INIT_BY_INSTRUMENT.get(inst_a, INITIAL_Z)
    # z_default_b = Z_INIT_BY_INSTRUMENT.get(inst_b, INITIAL_Z)
    # z_defaults = [(a + b) / 2.0 for a, b in zip(z_default_a, z_default_b)]

    z_source_key = (inst_a, inst_b)
    if st.session_state.get("z_source_key") != z_source_key:
        for i in range(len(z_defaults)):
            st.session_state[f"z_{i}"] = z_defaults[i]
        st.session_state["z_source_key"] = z_source_key

    z_values = []
    for i in range(4):
        val = st.slider(
            f"Z{i}",
            -2.0,
            2.0,
            st.session_state.get(f"z_{i}", z_defaults[i]),
            step=0.001,
            key=f"z_{i}",
        )
        z_values.append(val)

    st.header("Ganancia")
    default_xmax = (XMAX_DEFAULTS[inst_a] + XMAX_DEFAULTS[inst_b]) / 2.0
    xmax = st.slider("Xmax", 0.0, 300.0, default_xmax, step=1.0)

# Área principal: generación de audio
try:
    modelo, hps = get_interpolated_model(
        inst_a, inst_b, beta_config, training_source, alpha
    )

    audio = generate_audio(modelo, hps, z_values, xmax, NUM_FRAMES)

    sample_rate = hps["target_sampling_rate"]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)

    if play_enabled:
        audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        audio_html = f"""
        <audio id="vae_audio" controls autoplay loop>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        <script>
            const audio = document.getElementById('vae_audio');
            if (audio) {{
                audio.play().catch(() => {{}});
            }}
        </script>
        """
        components.html(audio_html, height=80)
    else:
        st.audio(buf, format="audio/wav", sample_rate=sample_rate)

except FileNotFoundError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Error generando audio: {e}")
