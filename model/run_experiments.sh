#!/bin/bash
set -euo pipefail

cd ~/Documents/2ASMRS
source .venv/bin/activate
cd model

# ── Parse flags ─────────────────────────────────────────────
SKIP_SLOW=0
for arg in "$@"; do
    case "$arg" in
        --skip-slow) SKIP_SLOW=1 ;;
        *)
            echo "Uso: bash run_experiments.sh [--skip-slow]"
            echo "  --skip-slow  Saltear experimentos lentos (UMAP, interpolación, morphing)"
            exit 1
            ;;
    esac
done

# ── Output dir ──────────────────────────────────────────────
NOW=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_${NOW}"
mkdir -p "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"

# ── Validar modelos requeridos ──────────────────────────────
MISSING=0

# Experimentos rápidos: arquitectura + comparación beta
REQUIRED_FAST=(
    "experiments_models/experiment_latent_dim_voice"
    "experiments_models/experiment_latent_dim_piano"
    "experiments_models/experiment_latent_dim_bass"
    "experiments_models/experiment_latent_dim_guitar"
    "experiments_models/base_model_beta_variation"
)

for d in "${REQUIRED_FAST[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Error: Falta '$d'"
        MISSING=1
    fi
done

# Experimentos lentos: UMAP, interpolación, morphing
if [ $SKIP_SLOW -eq 0 ]; then
    REQUIRED_SLOW=(
        "inference_models/base_model/base_model_beta_0.001/version_0"
        "inference_models/base_model/base_model_no_beta/version_0"
        "experiments_models/experiment_latent_dim_piano/beta_vae_latentdim_8/version_0"
        "experiments_models/experiment_latent_dim_piano/vae_latentdim_3/version_0"
    )

    INSTRUMENTS=(piano guitar voice bass)
    for instr in "${INSTRUMENTS[@]}"; do
        REQUIRED_SLOW+=(
            "inference_models/instruments_from_checkpoint/${instr}_from_checkpoint_beta_0.001"
            "inference_models/instruments_from_checkpoint/${instr}_from_checkpoint_no_beta"
            "inference_models/instruments_from_scratch/${instr}_from_scratch_beta_0.001"
            "inference_models/instruments_from_scratch/${instr}_from_scratch_no_beta"
        )
    done

    for d in "${REQUIRED_SLOW[@]}"; do
        if [ ! -d "$d" ]; then
            echo "Error: Falta '$d'"
            MISSING=1
        fi
    done
fi

if [ $MISSING -eq 1 ]; then
    echo "Faltan modelos requeridos. Abortando."
    exit 1
fi

# ════════════════════════════════════════════════════════════
# EXPERIMENTOS RÁPIDOS (siempre corren)
# ════════════════════════════════════════════════════════════

# 1) Arquitectura: error de validación vs espacio latente por instrumento
#    (experiments.md 1a, 1b, 2b, 2c)
echo ""
echo "═══ [1/6] Gráficos de arquitectura ═══"
python experiments/graficos_arquitectura.py \
    --output_dir "$OUTPUT_DIR/01_arquitectura"

# 2) Comparación beta-VAE: evolución del error de reconstrucción y KL
#    (experiments.md 2a)
echo ""
echo "═══ [2/6] Comparación beta-VAE ═══"
python experiments/graficos_comparacion_beta-vae.py \
    --output_dir "$OUTPUT_DIR/02_comparacion_beta_vae"

# ════════════════════════════════════════════════════════════
# EXPERIMENTOS LENTOS (se saltean con --skip-slow)
# ════════════════════════════════════════════════════════════

if [ $SKIP_SLOW -eq 1 ]; then
    echo ""
    echo "═══ Salteando experimentos lentos (--skip-slow) ═══"
else
    # 3) UMAP: análisis del espacio latente
    #    (experiments.md 3a)
    echo ""
    echo "═══ [3/6] UMAP: comparación de trayectorias ═══"
    python experiments/umap_experiment.py run_comparison \
        --output_img_path "$OUTPUT_DIR/03_umap/comparison_trajectory_piano.png"

    echo ""
    echo "═══ [3/6] UMAP: clustering de instrumentos en base model (β=0) ═══"
    python experiments/umap_experiment.py run_model_base_comparison \
        --base_model_path inference_models/base_model/base_model_no_beta/version_0 \
        --output_img_path "$OUTPUT_DIR/03_umap/base_model_no_beta_instruments_umap.png" \
        --run_label "Base Model (β=0)"

    echo ""
    echo "═══ [3/6] UMAP: clustering de instrumentos en base model (β=0.001) ═══"
    python experiments/umap_experiment.py run_model_base_comparison \
        --base_model_path inference_models/base_model/base_model_beta_0.001/version_0 \
        --output_img_path "$OUTPUT_DIR/03_umap/base_model_beta_instruments_umap.png" \
        --run_label "Base Model (β=0.001)"

    # 4) Interpolación: similitud de reconstrucción vs alfa
    #    (experiments.md 4a, 4b)
    echo ""
    echo "═══ [4/6] Interpolación (esto puede tardar 30+ minutos) ═══"
    OUTPUT_DIR="$OUTPUT_DIR/04_interpolacion" python experiments/interpolate.py

    # 5) Audio morphing: transición suave entre instrumentos
    echo ""
    echo "═══ [5/6] Audio morphing ═══"
    python experiments/audio_morphing.py
    # Mover outputs al directorio de resultados
    if [ -d "audio_morphing_output_griffinlim" ]; then
        mv audio_morphing_output_griffinlim "$OUTPUT_DIR/05_audio_morphing"
    fi

    # 6) Interpolación small: generación de audios interpolados por pares
    echo ""
    echo "═══ [6/6] Interpolación small (cross-interpolation) ═══"
    python experiments/interpolate_small.py
fi

# ════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "Experimentos finalizados."
echo "Resultados en: $OUTPUT_DIR"
echo "════════════════════════════════════════"
