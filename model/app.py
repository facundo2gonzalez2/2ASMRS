import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

from experiments.interpolate import interpolar_vae
from scripts.vae_predict import predict_audio
from VariationalAutoEncoder import VariationalAutoEncoder

# --- Constantes ---

INSTRUMENTS = ["piano", "guitar", "voice", "bass"]
XMAX_DEFAULTS = {
    "piano": 150.0,
    "guitar": 150.0,
    "voice": 140.0,
    "bass": 120.0,
}
# Presets de Z por configuración "{training_source}_{beta_config}".
# Sin beta el espacio latente no está centrado en 0 y cada modelo ocupa una región distinta.
# checkpoint_beta_0.001: media del encoder (scripts/sample_z_init.py). Resto: medoides de clusters (KMeans)
# de frames reales con energía alta y buena reconstrucción.
Z_INIT_BY_CONFIG = {
    "checkpoint_beta_0.001": {
        "piano": [0.00015, 0.10694, -0.02233, -0.09316],
        "guitar": [-0.00189, 0.17932, -0.11366, -0.21192],
        "voice": [-0.02621, 0.05876, 0.19291, 0.61994],
        "bass": [-0.0036, 0.14141, -0.00059, 0.06074],
    },
    "scratch_beta_0.001": {
        "piano": [1.2668, 1.4984, -0.3676, -0.0029],
        "guitar": [1.2152, 0.886, 0.0014, -0.0041],
        "voice": [0.0003, 0.05, 0.1004, 1.6939],
        "bass": [0.4265, -0.0011, 0.0034, -0.3605],
    },
    "checkpoint_no_beta": {
        "piano": [5.8271, 4.5897, 7.4451, -5.1208],
        "guitar": [-5.5814, 8.5572, 18.0094, -6.4851],
        "voice": [5.2364, 3.3322, 7.2178, -4.9473],
        "bass": [-7.512, 6.9719, 19.02, -5.975],
    },
    "scratch_no_beta": {
        "piano": [10.918, 2.5677, 5.5528, 9.5771],
        "guitar": [1.7153, 2.4796, 2.5315, -1.7724],
        "voice": [0.7347, 2.3654, 0.1631, 0.5248],
        "bass": [-3.5284, -1.6832, 3.3646, 1.8979],
    },
}
# Rango (min, max) de cada slider Z por configuración: percentiles 1-99 del encoder sobre los 4 instrumentos + margen.
Z_RANGE_BY_CONFIG = {
    "checkpoint_beta_0.001": [(-3.0, 3.0)] * 4,
    "scratch_beta_0.001": [(-3.0, 3.0)] * 4,
    "checkpoint_no_beta": [(-13.0, 11.0), (-1.0, 18.0), (2.0, 24.0), (-21.0, 0.0)],
    "scratch_no_beta": [(-8.0, 17.0), (-4.0, 6.0), (-3.0, 10.0), (-6.0, 14.0)],
}

NUM_FRAMES = 64


# --- Carga y cache de modelos ---


@st.cache_resource
def load_model(instrument_key: str, beta_config: str, training_source: str):
    """Carga un modelo VAE desde checkpoint. Resultado cacheado por Streamlit."""
    model_dir = Path(
        "model",
        "inference_models",
        f"instruments_from_{training_source}",
        f"{instrument_key}_from_{training_source}_{beta_config}",
        "version_0",
    )

    if not model_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del modelo: {model_dir}")

    ckpt_files = list(Path(model_dir, "checkpoints").glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No se encontraron checkpoints en: {model_dir / 'checkpoints'}")
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
        predicted_specgram = torch.nan_to_num(predicted_specgram, nan=0.0, posinf=xmax, neginf=0.0)
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


# --- Callbacks ---


def get_config(use_beta: bool, use_checkpoint: bool):
    beta_config = "beta_0.001" if use_beta else "no_beta"
    training_source = "checkpoint" if use_checkpoint else "scratch"
    return beta_config, training_source


def set_z_for_instrument(instrument_name, config_key):
    """Setea los valores de Z en session_state para un instrumento dado."""
    z_vals = Z_INIT_BY_CONFIG[config_key][instrument_name]
    for i, val in enumerate(z_vals):
        st.session_state[f"z_{i}"] = val


def on_config_change():
    """Al cambiar de modelo, Z vuelve al preset del instrumento A (cada config ocupa otra región del espacio latente)."""
    beta_config, training_source = get_config(st.session_state["use_beta"], st.session_state["use_checkpoint"])
    set_z_for_instrument(st.session_state.get("inst_a", "piano"), f"{training_source}_{beta_config}")


# --- UI ---

st.set_page_config(page_title="VAE Audio Interpolator", layout="wide")
st.title("VAE Audio Interpolator")

# Sidebar: solo configuración técnica
with st.sidebar:
    st.header("Configuración")
    use_beta = st.toggle("Beta encoding (beta_0.001)", value=True, key="use_beta", on_change=on_config_change)
    use_checkpoint = st.toggle("Desde checkpoint", value=True, key="use_checkpoint", on_change=on_config_change)

    beta_config, training_source = get_config(use_beta, use_checkpoint)
    config_key = f"{training_source}_{beta_config}"

# Main: Instrumentos y alpha
col_inst_a, col_inst_b, col_alpha = st.columns([1, 1, 2])
with col_inst_a:
    inst_a = st.selectbox("Instrumento A", INSTRUMENTS, index=0, key="inst_a")
with col_inst_b:
    inst_b = st.selectbox("Instrumento B", INSTRUMENTS, index=1)
with col_alpha:
    if inst_a == inst_b:
        st.info("Mismo instrumento. Alpha sin efecto.")
    alpha = st.slider("Alpha", 0.0, 1.0, 0.5, step=0.01)

# Main: Espacio latente (Z)
st.subheader("Espacio latente (Z)")

# Botones quick-set
btn_cols = st.columns(len(INSTRUMENTS))
for idx, instrument in enumerate(INSTRUMENTS):
    with btn_cols[idx]:
        st.button(
            instrument.capitalize(),
            on_click=set_z_for_instrument,
            args=(instrument, config_key),
            key=f"btn_z_{instrument}",
        )

# Inicializar Z en session_state si es la primera vez
if "z_0" not in st.session_state:
    set_z_for_instrument("piano", config_key)

# Sliders de Z
z_values = []
for i, (z_min, z_max) in enumerate(Z_RANGE_BY_CONFIG[config_key]):
    st.session_state[f"z_{i}"] = min(max(st.session_state[f"z_{i}"], z_min), z_max)
    val = st.slider(
        f"Z{i}",
        z_min,
        z_max,
        step=0.001,
        key=f"z_{i}",
    )
    z_values.append(val)

# Main: Ganancia
st.subheader("Ganancia")
default_xmax = (XMAX_DEFAULTS[inst_a] + XMAX_DEFAULTS[inst_b]) / 2.0
xmax = st.slider("Xmax", 0.0, 300.0, default_xmax, step=1.0)

# Main: Play y audio
play_enabled = st.toggle("Play", value=st.session_state.get("play_enabled", False))
st.session_state["play_enabled"] = play_enabled

try:
    modelo, hps = get_interpolated_model(inst_a, inst_b, beta_config, training_source, alpha)

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
