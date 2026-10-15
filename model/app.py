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

MODEL_DIR = Path(__file__).resolve().parent
INSTRUMENTS = ["piano", "guitar", "voice", "bass"]
# Configuraciones "{training_source}_{beta_config}" con la nomenclatura de la tesis.
CONFIGS = {
    "checkpoint_beta_0.001": "Checkpoint-β-VAE",
    "scratch_beta_0.001": "Scratch-β-VAE",
    "checkpoint_no_beta": "Checkpoint-AE",
    "scratch_no_beta": "Scratch-AE",
}
# Presets de Z por configuración.
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

NUM_FRAMES = 128  # ~3 s: un loop más largo hace menos notorio el corte al reiniciar
FADE_SECONDS = 0.01  # fade in/out para que el loop no haga click
# El nivel de salida varía ~60 dB entre configuraciones y valores de α: normalizar evita que el volumen sesgue
# la comparación. El tope de ganancia evita convertir un audio casi mudo en ruido fuerte.
TARGET_RMS_DB = -20.0
MAX_GAIN_DB = 60.0
# Barrido: ~1 s de A puro → transición de ~5 s en SWEEP_STEPS pasos de α → ~1 s de B puro.
SWEEP_HOLD_FRAMES = 43
SWEEP_STEPS = 21
SWEEP_STEP_FRAMES = 10


# --- Carga de modelos ---


@st.cache_resource(show_spinner=False)
def load_model(instrument_key: str, config_key: str):
    """Carga un modelo VAE desde checkpoint. Resultado cacheado por Streamlit."""
    training_source, beta_config = config_key.split("_", 1)
    model_dir = Path(
        MODEL_DIR,
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


@st.cache_resource(show_spinner="Cargando modelos...")
def warmup():
    """Precarga todos los modelos y corre un Griffin-Lim de prueba: el primero tarda ~10× más que los siguientes."""
    hps = None
    for config_key in CONFIGS:
        for instrument in INSTRUMENTS:
            try:
                _, hps = load_model(instrument, config_key)
            except FileNotFoundError:
                pass
    if hps is not None:
        predict_audio(
            predicted_specgram=torch.zeros(NUM_FRAMES, hps["win_length"] // 2 + 1),
            hps=hps,
            phase_option="griffinlim",
            frames=NUM_FRAMES,
            return_audio=True,
        )


def interpolate_models(inst_a_key: str, inst_b_key: str, config_key: str, alpha: float):
    """Interpola dos modelos VAE con SLERP (~0.05 s). Sin cache: se cachea el audio resultante."""
    model_a, hps_a = load_model(inst_a_key, config_key)

    # Si es el mismo instrumento, devolver modelo A directamente
    if inst_a_key == inst_b_key:
        return model_a, hps_a

    model_b, _ = load_model(inst_b_key, config_key)
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


def get_xmax(inst_a_key: str, inst_b_key: str, config_key: str, alpha: float, contrast: float):
    """Xmax de entrenamiento interpolado entre A y B (como en audio_morphing.py), escalado por el contraste."""
    _, hps_a = load_model(inst_a_key, config_key)
    _, hps_b = load_model(inst_b_key, config_key)
    return ((1.0 - alpha) * hps_a["Xmax"] + alpha * hps_b["Xmax"]) * contrast


# --- Generación de audio ---


def decode(model, z_values, xmax, num_frames):
    """Decodifica un Z repetido num_frames veces a un espectrograma en dB."""
    z = torch.tensor([z_values], dtype=torch.float32).repeat(num_frames, 1)

    with torch.no_grad():
        predicted_specgram = model.decoder(z) * xmax
        predicted_specgram = torch.nan_to_num(predicted_specgram, nan=0.0, posinf=xmax, neginf=0.0)
        return torch.clamp(predicted_specgram, min=0.0, max=xmax)


def normalize_loudness(audio):
    """Normaliza a TARGET_RMS_DB (ganancia limitada a MAX_GAIN_DB). Devuelve el audio y su nivel RMS original."""
    rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-12))
    audio = audio * 10 ** (min(TARGET_RMS_DB - rms_db, MAX_GAIN_DB) / 20)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio = audio * 0.98 / peak
    return audio, rms_db


def apply_fades(audio, sample_rate):
    n = min(int(FADE_SECONDS * sample_rate), len(audio) // 2)
    ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)
    audio = audio.copy()
    audio[:n] *= ramp
    audio[len(audio) - n :] *= ramp[::-1]
    return audio


def spectrogram_to_wav(predicted_specgram, hps):
    """Griffin-Lim (fase aleatoria si falla) → normalización → fades → WAV.

    Devuelve (wav_bytes, nivel RMS original en dBFS, si se usó fase aleatoria)."""
    used_fallback = False
    try:
        audio = predict_audio(
            predicted_specgram=predicted_specgram,
            hps=hps,
            phase_option="griffinlim",
            frames=predicted_specgram.shape[0],
            return_audio=True,
        )
    except Exception:
        used_fallback = True
        audio = predict_audio(
            predicted_specgram=predicted_specgram,
            hps=hps,
            phase_option="random",
            frames=predicted_specgram.shape[0],
            return_audio=True,
        )

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio, rms_db = normalize_loudness(audio)
    audio = apply_fades(audio, hps["target_sampling_rate"])

    buf = io.BytesIO()
    sf.write(buf, audio, hps["target_sampling_rate"], format="WAV")
    return buf.getvalue(), rms_db, used_fallback


# Cachear el audio (y no solo el modelo) hace que un rerun sin cambios devuelva los mismos bytes:
# el reproductor no se reinicia, y Griffin-Lim (que arranca de fase aleatoria) no da un audio distinto cada vez.
@st.cache_data(max_entries=64, show_spinner=False)
def render_audio(inst_a_key, inst_b_key, config_key, alpha, z_values, contrast):
    """Audio de ~3 s con Z fijo y los pesos interpolados en α."""
    model, hps = interpolate_models(inst_a_key, inst_b_key, config_key, alpha)
    xmax = get_xmax(inst_a_key, inst_b_key, config_key, alpha, contrast)
    return spectrogram_to_wav(decode(model, z_values, xmax, NUM_FRAMES), hps)


@st.cache_data(max_entries=32, show_spinner=False)
def render_sweep(inst_a_key, inst_b_key, config_key, z_a, z_b, contrast):
    """Barrido de α 0→1: A puro → transición (SLERP de pesos, Z interpolado de z_a a z_b) → B puro."""
    segments = (
        [(0.0, SWEEP_HOLD_FRAMES)]
        + [(float(alpha), SWEEP_STEP_FRAMES) for alpha in np.linspace(0.0, 1.0, SWEEP_STEPS)]
        + [(1.0, SWEEP_HOLD_FRAMES)]
    )
    specgrams = []
    for alpha, num_frames in segments:
        model, hps = interpolate_models(inst_a_key, inst_b_key, config_key, alpha)
        z = [(1.0 - alpha) * a + alpha * b for a, b in zip(z_a, z_b)]
        xmax = get_xmax(inst_a_key, inst_b_key, config_key, alpha, contrast)
        specgrams.append(decode(model, z, xmax, num_frames))
    return spectrogram_to_wav(torch.cat(specgrams), hps)


def show_audio(render_fn, *args, autoplay_loop=False):
    """Renderiza el audio con su nivel original; los errores se muestran en pantalla sin cortar la app."""
    try:
        wav, rms_db, used_fallback = render_fn(*args)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error generando audio: {e}")
        return

    if autoplay_loop:
        # st.audio(autoplay=True, loop=True) no sirve acá: no se detiene al apagar Play y no vuelve a
        # reproducir solo un audio que ya sonó en la sesión (p. ej. volver al preset anterior).
        audio_b64 = base64.b64encode(wav).decode("ascii")
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
        st.audio(wav, format="audio/wav")
    st.caption(f"Nivel original: {rms_db:.0f} dBFS RMS (normalizado a {TARGET_RMS_DB:.0f})")
    if used_fallback:
        st.warning("Griffin-Lim falló: este audio usa fase aleatoria.")


# --- Callbacks ---


def set_z_for_instrument(instrument_name, config_key):
    """Setea los valores de Z en session_state para un instrumento dado."""
    z_vals = Z_INIT_BY_CONFIG[config_key][instrument_name]
    for i, val in enumerate(z_vals):
        st.session_state[f"z_{i}"] = val


def on_config_change():
    """Al cambiar de modelo, Z vuelve al preset del instrumento A (cada config ocupa otra región del espacio latente)."""
    set_z_for_instrument(st.session_state.get("inst_a", "piano"), st.session_state["config"])


def request_sweep(state_key, params):
    """Marca qué barrido mostrar y apaga el loop para que no se superpongan."""
    st.session_state[state_key] = params
    st.session_state["play_enabled"] = False


# --- UI ---

st.set_page_config(page_title="VAE Audio Interpolator", layout="wide")
st.title("VAE Audio Interpolator")
warmup()

with st.sidebar:
    st.header("Avanzado")
    contrast = st.slider(
        "Contraste (× Xmax de entrenamiento)",
        0.5,
        2.0,
        1.45,
        step=0.05,
        help="Escala el rango en dB del espectrograma decodificado. No cambia el volumen: la salida se normaliza.",
    )

# Main: Instrumentos y alpha
col_inst_a, col_inst_b, col_alpha = st.columns([1, 1, 2])
with col_inst_a:
    inst_a = st.selectbox("Instrumento A", INSTRUMENTS, index=0, key="inst_a")
with col_inst_b:
    inst_b = st.selectbox("Instrumento B", INSTRUMENTS, index=1)
with col_alpha:
    alpha = st.slider("Alpha", 0.0, 1.0, 0.0, step=0.01)
    if inst_a == inst_b:
        st.info("Mismo instrumento. Alpha sin efecto.")
same_instrument = inst_a == inst_b

# Con on_change="rerun", .open permite generar solo los audios de la pestaña visible.
tab_explore, tab_compare = st.tabs(["Explorar", "Comparar configuraciones"], key="view", on_change="rerun")

with tab_explore:
    config_key = st.radio(
        "Configuración",
        list(CONFIGS),
        format_func=CONFIGS.get,
        horizontal=True,
        key="config",
        on_change=on_config_change,
    )

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
        set_z_for_instrument(inst_a, config_key)

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
    z_values = tuple(z_values)

    # Fuera de la pestaña visible no se renderiza el reproductor, así el loop deja de sonar al cambiar de pestaña.
    if tab_explore.open:
        col_play, col_audio = st.columns([1, 5])
        with col_play:
            play_enabled = st.toggle("Play (loop)", key="play_enabled")
        with col_audio:
            show_audio(render_audio, inst_a, inst_b, config_key, alpha, z_values, contrast, autoplay_loop=play_enabled)

        st.subheader("Barrido α 0→1")
        st.caption("Z fijo en los valores de los sliders; los pesos van de A a B con SLERP.")
        sweep_params = (inst_a, inst_b, config_key, z_values, z_values, contrast)
        st.button(
            "Generar barrido",
            on_click=request_sweep,
            args=("explore_sweep", sweep_params),
            disabled=same_instrument,
            key="btn_explore_sweep",
        )
        if not same_instrument and st.session_state.get("explore_sweep") == sweep_params:
            with st.spinner("Generando barrido..."):
                show_audio(render_sweep, *sweep_params)

with tab_compare:
    if tab_compare.open:
        st.caption(
            "Cada configuración usa sus propios presets de Z, interpolados con α: Z = (1−α)·Z_A + α·Z_B. "
            "El mismo Z no es comparable entre configuraciones (sin β el espacio latente no está centrado en 0)."
        )
        compare_params = (inst_a, inst_b, contrast)
        st.button(
            "Generar barridos α 0→1",
            on_click=request_sweep,
            args=("compare_sweep", compare_params),
            disabled=same_instrument,
            key="btn_compare_sweep",
        )
        show_sweeps = not same_instrument and st.session_state.get("compare_sweep") == compare_params

        cells = st.columns(2) + st.columns(2)
        for cell, (cfg, label) in zip(cells, CONFIGS.items()):
            with cell:
                st.markdown(f"**{label}**")
                z_a, z_b = tuple(Z_INIT_BY_CONFIG[cfg][inst_a]), tuple(Z_INIT_BY_CONFIG[cfg][inst_b])
                z_alpha = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(z_a, z_b))
                st.caption(f"α = {alpha:.2f}")
                show_audio(render_audio, inst_a, inst_b, cfg, alpha, z_alpha, contrast)
                if show_sweeps:
                    st.caption("Barrido α 0→1")
                    with st.spinner("Generando barrido..."):
                        show_audio(render_sweep, inst_a, inst_b, cfg, z_a, z_b, contrast)
