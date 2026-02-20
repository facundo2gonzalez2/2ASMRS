#!/bin/bash


cd ~/Documents/2ASMRS
source .venv/bin/activate
cd model

NOW=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_${NOW}"


# 1) Gráficos sobre arquitectura de los modelos

# Asegurarse de que existan los modelos
MODELS=(
    "experiment_latent_dim_voice_small"
    "experiment_latent_dim_piano_small"
    "experiment_latent_dim_bass_small"
    "experiment_latent_dim_guitar_small"
    "experiment_latent_dim_beta_voice_small"
    "experiment_latent_dim_beta_piano_small"
    "experiment_latent_dim_beta_bass_small"
    "experiment_latent_dim_beta_guitar_small"
)

MISSING_MODELS=0
for model_dir in "${MODELS[@]}"; do
    if [ ! -d "$model_dir" ]; then
        echo "Error: Falta el modelo/directorio '$model_dir'"
        MISSING_MODELS=1
    fi
done

if [ $MISSING_MODELS -eq 1 ]; then
    echo "Faltan algunos modelos exigidos por el script. Abortando ejecución."
    exit 1
fi

# Generar gráficos
python graficos_arquitectura.py --output_dir $OUTPUT_DIR/graficos_arquitectura


# 2) Comparación beta-vae
python graficos_comparacion_beta-vae.py --output_dir $OUTPUT_DIR/graficos_comparacion_beta-vae
