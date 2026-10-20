"""Genera fixtures de referencia para los tests de web/test (correr desde la raíz con el venv del proyecto).

- onnx_reference.json: salida de interpolar_vae(...).decoder(z) para algunos pares/α, para comparar con ORT Web.
- stft_target.f32: magnitudes STFT (Hann 2048 / hop 512, sin center) de un audio real, para el test de fase.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))

from audio_utils import get_waveform  # noqa: E402
from experiments.interpolate import interpolar_vae  # noqa: E402
from VariationalAutoEncoder import VariationalAutoEncoder  # noqa: E402
from web_export import load_hparams  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures"


def load(run):
    group = "instruments_from_" + ("checkpoint" if "checkpoint" in run else "scratch")
    d = ROOT / "model" / "inference_models" / group / run / "version_0"
    hps = load_hparams(d)
    ckpt = next((d / "checkpoints").glob("*.ckpt"))
    model = VariationalAutoEncoder(hps["encoder_layers"], hps["decoder_layers"], hps["latent_dim"], checkpoint_path=ckpt)
    return model.eval(), hps


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    z = torch.randn(4, 4)
    cases = []
    for run_a, run_b in [
        ("piano_from_checkpoint_beta_0.001", "guitar_from_checkpoint_beta_0.001"),
        ("voice_from_scratch_no_beta", "bass_from_checkpoint_no_beta"),
    ]:
        (ma, hps), (mb, _) = load(run_a), load(run_b)
        for alpha in (0.0, 0.35, 1.0):
            interp = interpolar_vae(ma, mb, alpha, hps["encoder_layers"], hps["decoder_layers"], hps["latent_dim"])
            with torch.no_grad():
                y = interp.eval().decoder(z)
            cases.append({"a": run_a, "b": run_b, "alpha": alpha, "z": z.tolist(), "y": y.tolist()})
    (OUT / "onnx_reference.json").write_text(json.dumps(cases))

    audio = next((ROOT / "model" / "data_instruments" / "piano").glob("*.mp3"))
    wav = get_waveform(audio, 22050)[22050 * 20 : 22050 * 26]
    spec = torch.stft(wav, 2048, 512, window=torch.hann_window(2048), center=False, return_complex=True).abs().T
    spec.numpy().astype("<f4").tofile(OUT / "stft_target.f32")
    print("frames", spec.shape)


if __name__ == "__main__":
    main()
