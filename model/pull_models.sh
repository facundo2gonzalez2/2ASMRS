#!/bin/bash

# Habilitar pipefail para que si falla la conexión SSH, la tubería (pipe) entera falle
set -o pipefail

# Definir los instrumentos en un array
instrumentos=("voice" "guitar" "bass" "piano")

# Diccionario para guardar el PID (Process ID) asociado a cada instrumento
declare -A pids

echo "🚀 Iniciando transferencias en paralelo..."
echo "----------------------------------------"

for inst in "${instrumentos[@]}"; do
    echo "Lanzando descarga de: $inst"
    
    # Rutas definidas para que quede más limpio
    remote_path="/home/fgbarnator/2ASMRS/model/experiments_models"
    remote_folder="experiment_latent_dim_beta_${inst}_small"
    local_dest="/home/facu/Documents/2ASMRS/model/experiments_models/experiment_latent_dim_${inst}"

    # Crear el directorio local destino si no existe (tar -C falla si no existe)
    mkdir -p "$local_dest"

    # Ejecutar el comando en background usando &
    ssh venus "tar -czf - -C '$remote_path' '$remote_folder'" | tar -xzf - -C "$local_dest" &
    
    # Guardar el PID de este proceso de fondo en el diccionario
    pids[$inst]=$!
done

echo "----------------------------------------"
echo "⏳ Todas las descargas están en curso. Esperando a que terminen..."
echo "----------------------------------------"

# Variable para saber si todo salió perfecto
todo_exito=true

# Recorrer los PIDs guardados y esperar a que termine cada uno
for inst in "${!pids[@]}"; do
    pid=${pids[$inst]}
    
    # El comando wait pausa el script hasta que el PID específico termine
    # y devuelve el exit code de ese proceso
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ÉXITO: '$inst' se transfirió correctamente."
    else
        echo "❌ ERROR: Falló la transferencia de '$inst' (Código de salida: $exit_code)."
        todo_exito=false
    fi
done

echo "----------------------------------------"
if $todo_exito; then
    echo "🎉 ¡Impecable! Todas las carpetas se trajeron y descomprimieron con éxito."
else
    echo "⚠️ Atención: Hubo errores en una o más transferencias. Revisá la salida de arriba."
fi