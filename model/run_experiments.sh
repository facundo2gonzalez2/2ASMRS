#!/bin/bash


cd ~/Documents/2ASMRS
source .venv/bin/activate
cd model

NOW=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_${NOW}"


# 1) Gráficos sobre arquitectura de los modelos

# Asegurarse de que existan los modelos
MODELS=(
    "experiments_models/experiment_latent_dim_voice"
    "experiments_models/experiment_latent_dim_piano"
    "experiments_models/experiment_latent_dim_bass"
    "experiments_models/experiment_latent_dim_guitar"
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
python experiments/graficos_arquitectura.py --output_dir $OUTPUT_DIR/graficos_arquitectura


# 2) Comparación beta-vae
python experiments/graficos_comparacion_beta-vae.py --output_dir $OUTPUT_DIR/graficos_comparacion_beta-vae
