# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**2ASMRS** (Audio Autoencoder for Sound Morphing in Realtime Synthesizer) — a Variational Autoencoder system that learns latent representations of audio spectrograms, enabling real-time sound morphing between instruments (piano, guitar, voice, bass). The Python model code lives in `model/`; the browser synthesizer lives in `web/` (see `web/README.md`); the C++ synthesizer plugin (`synthetizer/`) and supplementary scripts (`extra/`) are not currently active.

The project backs an undergraduate thesis. LaTeX sources live in `model/docs/plan-de-tesis/` (thesis plan) and `model/docs/tesis/` (thesis document).

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

# Train the full VAE pipeline (what main() currently runs: β-variation + latent-dim sweep on instruments)
cd model && python run_vae.py

# Train individual experiments via fire CLI (names match run_vae.py functions)
cd model && python run_vae.py train_base_model --beta 0.001
cd model && python run_vae.py train_instruments_models --from_checkpoint True --beta 0.001
cd model && python run_vae.py experiment_base_model_beta_variation
cd model && python run_vae.py experiment_latent_dim_instruments --beta 0
cd model && python run_vae.py experiment_latent_dim_base_model --beta 0

# Run the Streamlit interactive UI (port 8501)
streamlit run model/app.py --server.enableCORS false --server.enableXsrfProtection false

# Monitor training
tensorboard --logdir=model/tb_logs_vae

# Run all experiment graphs (requires pre-trained models). --skip-slow omits UMAP/interpolation/morphing.
cd model && bash run_experiments.sh [--skip-slow]

# Pull pre-trained latent-dim experiment models from remote host `venus` via SSH
cd model && bash pull_models.sh

# Browser synth: export checkpoints to web/public/models/, then run the Vite dev server
.venv/bin/python model/scripts/export_web_models.py
cd web && npm install && npm run dev
cd web && npm test   # ONNX Web vs PyTorch parity, phase-reconstruction quality, offline WAV render
```

There is no Python test suite (`web/` has `npm test`). Formatting is handled by Black (config in `pyproject.toml`: line-length 120, target py312). VSCode settings in `.vscode/settings.json` use Pyright in basic mode and Black on save.

## Directory Structure

```
model/
├── app.py                     # Streamlit interactive UI
├── audio_comparator.py        # FAD-based audio quality assessment
├── audio_utils.py             # STFT/ISTFT, spectrogram utilities
├── run_vae.py                 # Main training orchestration (fire CLI)
├── VariationalAutoEncoder.py  # VAE model definition (PyTorch Lightning)
├── web_export.py              # InterpolatedDecoder (SLERP in forward) + ONNX/weights export for web/
├── utils.py                   # Shared helpers
├── run_experiments.sh         # Run all experiment graph scripts
├── pull_models.sh             # Pull pre-trained experiment models from remote
├── train.sh                   # Training shell script
├── requirements.txt
├── docs/                      # Project documentation
│   ├── datasets.md            # Dataset descriptions and paths
│   ├── experiments.md         # Thesis experiments & graphs reference
│   ├── model-nomenclature.md  # Model naming conventions
│   ├── plan-de-tesis/         # LaTeX sources for the thesis plan
│   └── tesis/                 # LaTeX sources for the thesis
├── experiments/               # Experiment & visualization scripts
│   ├── interpolate.py         # SLERP/linear interpolation + FAD metrics & caching
│   ├── interpolate_small.py   # SLERP interpolation (lightweight, cross-interpolation audios)
│   ├── audio_morphing.py      # Audio-to-audio morphing (encode → interpolate models → decode)
│   ├── crossfade_morphing.py  # Cross-fade baseline vs. morphing on the same two audios (WAVs + spectrogram plot)
│   ├── crossfade_fad.py       # FAD-similarity curves vs α: spectrogram cross-fade baseline vs. VAE morphing
│   ├── crossfade_fad_calibration.py  # FAD calibration on plain waveform cross-fades of original audio (no models)
│   ├── umap_experiment.py     # UMAP latent space visualization
│   ├── graficos_arquitectura.py          # Architecture / latent-dim comparison plots
│   └── graficos_comparacion_beta-vae.py  # β-VAE comparison plots
├── scripts/                   # Utility & data-processing scripts
│   ├── vae_predict.py         # Run inference with trained VAE (supports multiple phase modes)
│   ├── sample_test_dataset.py # Generate test dataset splits
│   ├── sample_z_init.py       # Compute Z init stats (mean/std) per instrument for app.py
│   ├── export_web_models.py   # Export checkpoints + manifest for the browser synth (web/public/models/)
│   ├── fad_similarity_plot.py # FAD similarity visualization
│   ├── process_dataset.py     # Dataset preprocessing
│   └── stero_to_mono.py       # Stereo-to-mono audio conversion
├── imgs/                      # Experiment result plots
├── umap_experiment/           # UMAP visualization outputs
├── inference_models/          # Models for inference
│   ├── base_model/
│   │   ├── base_model_no_beta/
│   │   └── base_model_beta_0.001/
│   ├── instruments_from_checkpoint/  # Per-instrument fine-tuned from base (committed)
│   │   └── {piano,guitar,voice,bass}_from_checkpoint_{no_beta,beta_0.001}/
│   └── instruments_from_scratch/     # Per-instrument trained from scratch (gitignored)
│       └── {piano,guitar,voice,bass}_from_scratch_{no_beta,beta_0.001}/
└── experiments_models/        # Models trained for experimentation (gitignored)
    ├── experiment_latent_dim_{piano,guitar,voice,bass}/  # Latent dim sweep per instrument
    ├── experiment_latent_dim_base_model/                 # Latent dim sweep on base model
    └── base_model_beta_variation/                        # β variation experiments
```

## Architecture

### Data Pipeline

Audio files → `audio_utils.get_spectrograms_from_audios()` (STFT → dB-normalized spectrograms + phase extraction) → normalized numpy arrays used as both input and target for the autoencoder.

Reconstruction: decoder output × Xmax → `audio_utils.spectrogram2audio()` → WAV/MP3. Four phase-reconstruction modes are supported and selected at inference time via `scripts/vae_predict.predict_audio(phase_option=...)`:
- `griffinlim` — Griffin-Lim iterative phase reconstruction (librosa).
- `pghi` — Phase Gradient Heap Integration via `tifresi.stft.GaussTF` (requires `ltfatpy`-compatible frame sizing; pads frames automatically).
- `random` — Random phase (fast, lossy).
- `pv` — Phase-vocoder-style phase (only used in vae_predict, not exposed from app/morphing).

### Model

`VariationalAutoEncoder` (PyTorch Lightning module in `VariationalAutoEncoder.py`):
- **VAEEncoder**: Linear layers with BatchNorm1d + ELU → two heads (mu, logvar) → reparameterization trick
- **Decoder**: Mirror architecture, Linear + BatchNorm1d + ELU; final layer is Linear + BatchNorm1d (no activation)
- Default architecture: input=1025 (win_length/2+1) → `(1024,512,256,128,64,32,16,8,4)` → latent_dim=4
- Loss: MSE reconstruction + β × KL divergence (β=0.001 is the standard config; when β=0, forward uses `z=mu` instead of sampling)
- `train_model()` handles DataLoader creation, TensorBoard logging, early stopping (patience=100 on `val_loss`), and checkpointing

### Training Strategy (run_vae.py)

The `fire` CLI exposes each top-level function directly. Key entry points:

- `train_base_model(beta, **kwargs)` — Train on all instruments combined. Logs under `inference_models/base_model/base_model_{beta_str}`.
- `train_instruments_models(from_checkpoint, beta, **kwargs)` — Per-instrument training; when `from_checkpoint=True` loads the matching `base_model_{beta_str}` checkpoint. Logs under `inference_models/instruments_from_{checkpoint,scratch}`.
- `experiment_base_model_beta_variation(kwargs)` — Base-model training for 5 betas: `[0.01, 0.001, 0.0001, 0.00001, 0]`.
- `experiment_latent_dim_instruments(beta, kwargs)` — Latent-dim sweep (2,3,4,6,8) × per-instrument × 5 rounds, using `data_instruments_small/`.
- `experiment_latent_dim_base_model(beta, kwargs)` — Same sweep but on the combined dataset.
- `*_with_full_latent_dim(...)` — Variants with a larger architecture (`(2048,…,8)`, latent_dim=8).

`main()` is what `python run_vae.py` invokes without args. It currently runs only `experiment_base_model_beta_variation` + `experiment_latent_dim_instruments(beta=0)`; other stages are commented out.

### Interpolation & Morphing

- `experiments/interpolate_small.py` — SLERP (Spherical Linear Interpolation) over model weights to smoothly morph between two trained instrument models. Falls back to LERP when weight vectors are near-parallel. Supports cross-interpolation audio generation.
- `experiments/interpolate.py` — Adds FAD-based similarity metrics, caching, and the `interpolar_vae(model_a, model_b, alpha, ..., interpolation_mode)` helper consumed by `app.py` and `audio_morphing.py`.
- `experiments/audio_morphing.py` — Encodes two real audio files, then produces a three-phase output: pure A → transition (blending both Z sequences and interpolating model weights with α going 0→1) → pure B. Writes WAV + YAML metadata into `audio_morphing_output_{mode}_{phase_reconstruction}/`. Config is inline at the top of `main()` (no CLI args).
- `experiments/crossfade_morphing.py` / `experiments/crossfade_fad.py` — Naive cross-fade baseline (mixing spectrograms instead of interpolating Z + weights). The first renders the same A→B timeline as `audio_morphing.py` with both methods; the second computes FAD similarity vs α against source/target/witness references over N segment pairs (outputs in `crossfade_*_output*/`). `experiments/crossfade_fad_calibration.py` drops the models entirely: it cross-fades original waveform segments `(1−α)·A + α·B` and measures FAD similarity against the original A/B/witness segments, to calibrate the metric (for the thesis section that introduces FAD).

### Streamlit App (app.py)

Interactive UI that loads two instrument models, interpolates them with SLERP at a configurable α, then generates audio from a 4D latent vector (Z0–Z3) controlled via sliders. Models are cached with `@st.cache_resource`. Audio is generated by repeating the Z vector across 64 frames through the decoder. Per-instrument Z-init presets (`Z_INIT_BY_INSTRUMENT`) come from `scripts/sample_z_init.py`.

### Audio Quality Assessment

`audio_comparator.py` uses the MERT-v1-95M transformer model to compute audio embeddings and Fréchet Audio Distance (FAD) between distributions.

### Browser Synthesizer (web/)

Vite + TypeScript, no UI framework. `model/web_export.py` defines `InterpolatedDecoder(state_a, state_b)` with `forward(z, alpha)` doing per-tensor SLERP exactly like `interpolar_vae`, and exports **one ONNX graph per architecture** whose weights are external data `A.bin`/`B.bin`; each checkpoint becomes one `weights/{id}.bin` with the same layout, so the browser pairs any two checkpoints via ONNX Runtime Web `externalData`. `train()` in `run_vae.py` auto-exports at the end of training; `scripts/export_web_models.py` (inline `roots` config) exports existing checkpoints. Output in `web/public/models/`, committed: it's what the Railway deploy serves (root directory `web/`, static Vite build). Tracked set = the 16 per-instrument models (from_checkpoint/from_scratch × β/no-β); base models are excluded from `roots`.

Runtime: `src/engine/synth.worker.ts` (ORT wasm decoder + `src/dsp/` phase reconstruction + overlap-add) → MessageChannel → `src/engine/player.worklet.ts` (block queue, volume, auto-gain). Phase modes in `src/dsp/`: RTISI-LA initialized with RTPGHI (default), RTPGHI alone, random (as in the C++ synth). UI state persists to localStorage (`src/state.ts`).

### Model Export

`VariationalAutoEncoder.export_decoder()` traces the decoder to TorchScript (`.pt`) with a companion `.json` config file (latent ranges, STFT params, Xmax) for the C++ synthesizer plugin.

### Datasets

See `model/docs/datasets.md` for full details.
- **`data_instruments/`** — Main training data with subdirectories per instrument (piano, guitar, voice, bass).
- **`data_test/`** — Test split generated by `scripts/sample_test_dataset.py`.
- **`data_instruments_small/`** — Smaller variant with the same instrument subdirectories, used by the latent-dim sweeps.

### Experiments (Thesis)

See `model/docs/experiments.md` for the full list. Numbered in `run_experiments.sh`:
1. Architecture / latent-dim — reconstruction error vs. latent space per instrument.
2. β-VAE comparison — reconstruction loss evolution for 5 β values; KL vs. latent dim; MSE vs. latent dim at β=0.001.
3. UMAP — visualizing clustering of instruments in the base model's latent space.
4. Interpolation — reconstruction similarity (cosine + FAD) vs. α for pairs of instruments; from-checkpoint vs. from-scratch and β vs. no-β comparisons.
5. Audio morphing — pair-wise audio transitions (output in `audio_morphing_output_*`).
6. Interpolation small — pair-wise interpolated audios for listening.

### Model Nomenclature

See `model/docs/model-nomenclature.md` for full details.

**Experiments (`model/experiments_models/`):**
- `base_model_beta_variation/base_model_beta_{b}` — 5 versions for b ∈ [0.01, 0.001, 0.0001, 0.00001, 0].
- `experiment_latent_dim_{instrument}/{beta_vae,vae}_latentdim_{n}` — Latent dim experiments per instrument (n ∈ {2,3,4,6,8}, 5 versions each).
- `experiment_latent_dim_base_model/{beta_vae,vae}_latentdim_{n}` — Same sweep on base model.

**Inference (`model/inference_models/`):**
- `base_model/base_model_no_beta` and `base_model/base_model_beta_0.001` — single version_0 each.
- `instruments_from_{checkpoint,scratch}/{instrument}_from_{checkpoint,scratch}_{beta_variant}` — per-instrument fine-tuned models.

## Key Conventions

- Scripts in `model/` use relative imports and expect `model/` as cwd. Scripts inside `experiments/` and `scripts/` inject `MODEL_DIR` into `sys.path` so they can be invoked from anywhere.
- Experiment scripts live in `model/experiments/`, utility scripts in `model/scripts/`.
- Documentation lives in `model/docs/`, experiment result images in `model/imgs/`.
- Training data goes in `model/data_instruments/{piano,voice,guitar,bass}/` (gitignored).
- Test data in `model/data_test/` and `model/data_test_gt/` (gitignored); small dataset in `model/data_instruments_small/` (gitignored).
- Checkpoints saved to `model/inference_models/instruments_from_checkpoint/` (committed) and `model/inference_models/instruments_from_scratch/` (gitignored).
- Experiment models saved to `model/experiments_models/` (gitignored).
- TensorBoard logs to `model/tb_logs_vae/` (gitignored).
- Exported TorchScript models to `model/exported_models_vae/` (gitignored).
- Audio-morphing outputs go to `model/audio_morphing_output_*/` (gitignored).
- Web-exported models go to `web/public/models/` (committed, served by the deploy); `web/node_modules/`, `web/dist/`, `web/test/out/` are gitignored.
- Hyperparameters stored in `hparams.yaml` alongside checkpoints; loaded via YAML for inference.
- Instrument directory and display names are identical (`piano`, `guitar`, `voice`, `bass`) — no aliasing.
- Experiment scripts use inline config at the top of `main()` rather than fire/argparse for per-run tweaks.
