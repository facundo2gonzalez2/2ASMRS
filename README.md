# Audio Autoencoder for Sound Morphing in Realtime Synthesizer

## LAMIR 2026 paper: *Timbre Morphing via Weight Interpolation of Specialized Variational Autoencoders*

- Paper source and figures: [`model/docs/Paper_Template_for_LAMIR_2024/`](model/docs/Paper_Template_for_LAMIR_2024/)
- Full thesis (Spanish), with every experiment and plot: [`model/docs/tesis/tesis.pdf`](model/docs/tesis/tesis.pdf)
- Trained per-instrument models (web export): [`web/public/models/`](web/public/models/); browser synthesizer: [`web/`](web/)
- Model code: [`model/`](model/) (training: `run_vae.py`, weight interpolation: `experiments/interpolate.py`)

### Audio examples

Continuous weight interpolation (α swept from 0 to 1):

- Voice → piano, Checkpoint-β-VAE: [listen](https://drive.google.com/file/d/1EYOZBdlWy43A6llsoD5yIvSeZali93yI/view?usp=sharing)
- Guitar → piano, Checkpoint-β-VAE: [listen](https://drive.google.com/file/d/1eSDCLBd_SoxXbaXlolphe5Y2nrI0whpW/view?usp=sharing)
- Piano → guitar, Scratch-β-VAE (contrast): [listen](https://drive.google.com/file/d/13rdDVCrwNFWcaPPINCZFoNJ-DAYwCm3v/view?usp=sharing)
- Piano → guitar, Scratch-AE (worst configuration): [listen](https://drive.google.com/file/d/14CRDhlVectmjLzDP5--KUlRY0CI5kBS7/view?usp=sharing)

Audio-to-audio morphing between real recordings (voice A → guitar B, 2 s A, 8 s transition, 2 s B):

- Original A (voice): [listen](https://drive.google.com/file/d/1_P2HaMeGV1i4mmaRes1mp8QyUxHjCl6v/view?usp=sharing)
- Original B (guitar): [listen](https://drive.google.com/file/d/1c5WpXuB0kv0syMvrudzjVG3S2fCLoaEj/view?usp=sharing)
- Waveform cross-fade (reference): [listen](https://drive.google.com/file/d/1yJyHLEhtkX4pwjxaaCggMm5hjcC4FZ2I/view?usp=sharing)
- Weight-interpolation morphing (Checkpoint-β-VAE, PGHI phase): [listen](https://drive.google.com/file/d/1V-eO7fWwcp6LsccxIHGB0uGlvY1LILmz/view?usp=sharing)

### Reproducing the paper's numbers

Both scripts only read existing outputs: the TensorBoard logs of `model/inference_models/` and the CSVs written by
`model/experiments/interpolate.py` (run it first if `model/interpolation_random_*` is missing).

```bash
.venv/bin/python model/scripts/summarize_interpolation_metrics.py   # Table 1 (from existing logs/CSVs)
.venv/bin/python model/scripts/plot_cross_transition.py             # Figure 2 (vector PDF)
```

# Install

Clone with submodules

```
git clone --recursive https://github.com/pabloriera/2ASMRS.git
```

# Model

## Train example:
```console
python run.py
```

## TensorBoard:
```console
tensorboard --logdir=logs
```
## Train on colab

[Colab](https://colab.research.google.com/drive/1VMalSDqbO-idkTdtX47v7qRcLDioZ5GT#scrollTo=4zlUqBjwMCeu)

# Synthesizer:

## Building:

Download the .zip file containing the libtorch (PyTorch C++ API) source and unzip in external folder

```console
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
```

### Linux dependencies

```
sudo apt update
sudo apt install libasound2-dev libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev  \
    libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev
```

### Build 
```console
cmake -Bbuild
cmake --build build --config Release
```

### Run
```console
build/AE_artefacts/Debug/AE test_torchscript.json
```
