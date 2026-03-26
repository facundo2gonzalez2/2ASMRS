# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**2ASMRS** (Audio Autoencoder for Sound Morphing in Realtime Synthesizer) — a Variational Autoencoder system that learns latent representations of audio spectrograms, enabling real-time sound morphing between instruments (piano, guitar, vocals, bass). The Python model code lives in `model/`; the C++ synthesizer plugin (`synthetizer/`) and supplementary scripts (`extra/`) are not currently active.

## Virtual Environment

The Python virtual environment is at `.venv/` in the project root. Always use it when running Python commands:

```bash
source .venv/bin/activate
# or call directly: .venv/bin/python
```

## Commands

All Python commands run from `model/` as the working directory using the project venv.

```bash
# Install dependencies
.venv/bin/pip install -r model/requirements.txt

# Train the full VAE pipeline (base models + per-instrument fine-tuning)
cd model && python run_vae.py

# Train individual experiments via fire CLI
cd model && python run_vae.py train_model_base --beta 0.001 --epochs 1000
cd model && python run_vae.py train_model_instruments --from_checkpoint True --beta 0.001

# Run the Streamlit interactive UI (port 8501)
streamlit run model/app.py --server.enableCORS false --server.enableXsrfProtection false

# Monitor training
tensorboard --logdir=model/tb_logs_vae

# Run all experiment graphs (requires pre-trained models)
cd model && bash run_experiments.sh
```

There is no test suite or linter configured.

## Directory Structure

```
model/
├── app.py                     # Streamlit interactive UI
├── audio_comparator.py        # FAD-based audio quality assessment
├── audio_utils.py             # STFT/ISTFT, spectrogram utilities
├── run_vae.py                 # Main training orchestration (fire CLI)
├── VariationalAutoEncoder.py  # VAE model definition (PyTorch Lightning)
├── utils.py                   # Shared helpers
├── run_experiments.sh          # Run all experiment graph scripts
├── train.sh                   # Training shell script
├── requirements.txt
├── docs/                      # Project documentation
│   ├── datasets.md            # Dataset descriptions and paths
│   └── model-nomenclature.md  # Model naming conventions
├── experiments/               # Experiment & visualization scripts
│   ├── interpolate.py         # SLERP interpolation + FAD metrics & caching
│   ├── interpolate_small.py   # SLERP interpolation (lightweight)
│   ├── umap_experiment.py     # UMAP latent space visualization
│   ├── graficos_arquitectura.py          # Architecture comparison plots
│   └── graficos_comparacion_beta-vae.py  # β-VAE comparison plots
├── scripts/                   # Utility & data-processing scripts
│   ├── vae_predict.py         # Run inference with trained VAE
│   ├── sample_test_dataset.py # Generate test dataset splits
│   ├── sample_z_init.py       # Sample initial latent vectors
│   ├── fad_similarity_plot.py # FAD similarity visualization
│   ├── process_dataset.py     # Dataset preprocessing
│   └── stero_to_mono.py       # Stereo-to-mono audio conversion
├── imgs/                      # Experiment result plots
├── umap_experiment/           # UMAP visualization outputs
├── inference_models/          # Models for inference
│   ├── base_model/            # Base models trained on all instruments
│   ├── instruments_from_checkpoint/  # Per-instrument fine-tuned from base (committed)
│   └── instruments_from_scratch/     # Per-instrument trained from scratch (gitignored)
└── experiments_models/        # Models trained for experimentation
    ├── experiment_latent_dim_*_small/       # Latent dim sweep per instrument
    ├── experiment_latent_dim_base_model/    # Latent dim sweep on base model
    └── base_model_beta_variation/           # β variation experiments
```

## Architecture

### Data Pipeline

Audio files → `audio_utils.get_spectrograms_from_audios()` (STFT → dB-normalized spectrograms + phase extraction) → normalized numpy arrays used as both input and target for the autoencoder.

Reconstruction: decoder output × Xmax → `audio_utils.spectrogram2audio()` (ISTFT with Griffin-Lim, random, or phase vocoder phases) → WAV/MP3.

### Model

`VariationalAutoEncoder` (PyTorch Lightning module in `VariationalAutoEncoder.py`):
- **VAEEncoder**: Linear layers with BatchNorm1d + ELU → two heads (mu, logvar) → reparameterization trick
- **Decoder**: Mirror architecture, Linear + BatchNorm1d + ELU (no activation on final layer)
- Default architecture: input=1025 (win_length/2+1) → `(1024,512,256,128,64,32,16,8,4)` → latent_dim=4
- Loss: MSE reconstruction + β × KL divergence (β=0.001 is the standard config)
- `train_model()` handles DataLoader creation, TensorBoard logging, early stopping, and checkpointing

### Training Strategy (run_vae.py)

Two-stage training orchestrated by `main()`:
1. **Base model**: Train on all instruments combined (`train_model_base`)
2. **Per-instrument fine-tuning**: Load base model checkpoint, fine-tune on single instrument (`train_model_instruments --from_checkpoint True`)

Both stages run with β=0 and β=0.001 variants. The `fire` CLI exposes each function directly.

### Interpolation

`experiments/interpolate_small.py` implements SLERP (Spherical Linear Interpolation) over model weights to smoothly morph between two trained instrument models. Falls back to LERP when weight vectors are near-parallel. `experiments/interpolate.py` extends this with FAD-based similarity metrics and caching.

### Streamlit App (app.py)

Interactive UI that loads two instrument models, interpolates them with SLERP at a configurable α, then generates audio from a 4D latent vector (Z0–Z3) controlled via sliders. Models are cached with `@st.cache_resource`. Audio is generated by repeating the Z vector across 64 frames through the decoder.

### Audio Quality Assessment

`audio_comparator.py` uses the MERT-v1-95M transformer model to compute audio embeddings and Fréchet Audio Distance (FAD) between distributions.

### Model Export

`export_decoder()` traces the decoder to TorchScript (`.pt`) with a companion `.json` config file (latent ranges, STFT params, Xmax) for the C++ synthesizer plugin.

### Datasets

See `model/docs/datasets.md` for full details. Three datasets exist:
- **`data_instruments/`** — Main training data with subdirectories per instrument (piano, guitar, voice, bass)
- **`data_test/`** — Test split generated by `scripts/sample_test_dataset.py`
- **`data_instruments_small/`** — Smaller variant with the same instrument subdirectories

### Model Nomenclature

See `model/docs/model-nomenclature.md` for full details. Key naming patterns:
- `inference_models/base_model/base_model_no_beta` / `inference_models/base_model/base_model_beta_{b}` — Base models trained on all instruments
- `experiments_models/base_model_beta_variation` — 5 versions for different β values
- `experiments_models/experiment_latent_dim_{instrument}_small` / `experiments_models/experiment_latent_dim_beta_{instrument}_small` — Latent dimension experiments per instrument (dims 2,3,4,6,8)
- `experiments_models/experiment_latent_dim_base_model` — Latent dimension experiments on base model
- `inference_models/instruments_from_checkpoint` / `inference_models/instruments_from_scratch` — Per-instrument fine-tuned models with `{instrument}_from_{strategy}_{beta_variant}` naming

## Key Conventions

- Scripts in `model/` use relative imports and expect `model/` as cwd
- Experiment scripts live in `model/experiments/`, utility scripts in `model/scripts/`
- Documentation lives in `model/docs/`, experiment result images in `model/imgs/`
- Training data goes in `model/data_instruments/{piano,voice,guitar,bass}/` (gitignored)
- Test data in `model/data_test/` (gitignored), small dataset in `model/data_instruments_small/` (gitignored)
- Checkpoints saved to `model/inference_models/instruments_from_checkpoint/` (committed) and `model/inference_models/instruments_from_scratch/` (gitignored)
- Experiment models saved to `model/experiments_models/` (gitignored)
- TensorBoard logs to `model/tb_logs_vae/` (gitignored)
- Exported TorchScript models to `model/exported_models_vae/` (gitignored)
- Hyperparameters stored in `hparams.yaml` alongside checkpoints; loaded via YAML for inference
- The app maps display name "vocals" to directory name "voice" via `INSTRUMENT_DIR_NAMES`
