# 2ASMRS web: sintetizador en el navegador

Sintetizador en tiempo real que corre todo en el browser: decoder VAE con **interpolación de pesos entre dos
checkpoints dentro del forward** (ONNX Runtime Web), reconstrucción de fase en tiempo real y overlap-add en un
AudioWorklet.

```bash
# 1. Exportar los checkpoints (una vez, y después de entrenar fuera de run_vae.py)
.venv/bin/python model/scripts/export_web_models.py

# 2. Correr la UI
cd web && npm install && npm run dev       # http://localhost:5173
npm run build                               # sitio estático en web/dist/ (base relativa)
npm test                                    # paridad ONNX vs PyTorch + calidad de fase + render a WAV
```

Requiere Node ≥ 18. `npm test` necesita los modelos exportados y escribe WAVs en `test/out/` para escuchar.
Los fixtures de `test/fixtures/` se regeneran con `.venv/bin/python web/test/make_fixtures.py`.

## Cómo funciona

```
UI (main thread) ──params──▶ Worker de síntesis ──bloques de hop──▶ AudioWorklet ──▶ salida
   pads Z, α, presets         ONNX Runtime Web (wasm)                 cola + volumen
   localStorage               |X| = 10^((y·Xmax + dbMin)/20)          auto-ganancia
                              fase (RTISI-LA / RTPGHI / aleatoria)    soft-clip
                              iFFT + Hann + overlap-add
```

- **Modelo** (`model/web_export.py`): `InterpolatedDecoder(state_a, state_b)` con `forward(z, alpha)`. Cada tensor
  se interpola con SLERP igual que `interpolar_vae` (w1(α)·A + w2(α)·B, fallback a LERP). Las capas lineales
  calculan w1·(A·x) + w2·(B·x) para no materializar la matriz interpolada. Error vs `interpolar_vae`: ~1e-5.
- **Un grafo por arquitectura, un `.bin` por checkpoint**: el ONNX referencia los pesos como datos externos
  `A.bin`/`B.bin`, y el browser les asigna los `.bin` de los dos checkpoints elegidos (`externalData`). Así se
  combina cualquier par sin exportar N² modelos. Los ángulos SLERP dependen solo de los pesos y ORT los pliega
  como constantes al crear la sesión (~200–600 ms). Por frame solo se computa lo que depende de α (~1–2 ms).
- **Fase**: RTISI-LA (Griffin-Lim en tiempo real con lookahead, inicializado con RTPGHI) por defecto; RTPGHI solo
  (sin latencia extra) y fase aleatoria (como el sintetizador C++) como alternativas. Consistencia espectral medida
  en `npm test`:

  | Método                       | audio real (piano) | espectro decodificado estático | costo/frame |
  | ---------------------------- | ------------------ | ------------------------------ | ----------- |
  | Fase aleatoria (C++)         | −3.6 dB            | −4.4 dB                        | 0.2 ms      |
  | RTPGHI                       | −21 dB             | −9.5 dB                        | 0.6 ms      |
  | RTISI-LA (lookahead 2, 8 it) | −29 dB             | −15 dB                         | ~5 ms       |
  | Griffin-Lim offline 32 it    | −23 dB             | −16.5 dB                       | —           |

- **Cambio de modelos sin cortes**: antes de crear la sesión nueva (que bloquea el worker) se mandan al worklet
  bloques extra generados con la sesión vieja.

## Agregar modelos

- Entrenando con `run_vae.py`: `train()` exporta el modelo solo al terminar (pesos + rangos de Z sobre todo el
  dataset). Recargá la página o tocá ↻ en la UI.
- Checkpoints existentes: agregar su directorio de logs a `roots` en `model/scripts/export_web_models.py` y
  correrlo. Los rangos de los pads y el Z inicial salen del encoder sobre algunos audios de `data_instruments/`
  (si no están, se usa el prior N(0, 1)).

Los modelos van a `web/public/models/` (commiteado, es lo que sirve el deploy): `manifest.json`, `graphs/{arch}.onnx` (~190 KB) y
`weights/{id}.bin` (~6.8 MB en float32). Solo se pueden interpolar checkpoints con la misma arquitectura y los
mismos parámetros de STFT; la UI deshabilita los incompatibles.

## UI

- **A / B**: checkpoints a interpolar (⇄ intercambia, ↻ relee el manifest). **α**: interpolación de pesos.
- **Pads XY**: Z0·Z1 y Z2·Z3 (dimensiones extra como sliders). Los círculos marcan el Z inicial de A y B; doble
  click vuelve a él. Flechas del teclado para ajuste fino (Shift = más fino).
- **Presets** (10): click carga, mantener apretado (650 ms) guarda/sobrescribe. Teclas `1`–`0` cargan,
  `Shift+n` guarda. Guardan Z, α y volumen.
- **Espacio**: play/stop. El estado completo (modelos, sliders, presets, ajustes) se guarda continuamente en
  `localStorage`; si estaba sonando al cerrar, el audio se reanuda con la primera interacción (los browsers no
  permiten arrancar audio sin un gesto del usuario).
- **Avanzado**: método de fase, lookahead/iteraciones de RTISI, contraste (× Xmax, default 1.45 como la demo de
  Streamlit), glide de Z/α, tamaño del buffer y auto-ganancia (normaliza a −20 dB RMS: el nivel cambia ~60 dB
  entre modelos y α).

## ONNX Runtime Web: limitaciones

- **No corre dentro del AudioWorklet** (API asíncrona, sin `fetch`/import dinámico en ese scope): la inferencia va
  en un Worker y le pasa bloques al worklet. Eso suma latencia: buffer (3 bloques ≈ 70 ms por defecto, ajustable)
  + lookahead de RTISI (2 frames ≈ 46 ms) + la salida de audio. Total ≈ 140 ms con los defaults; con RTPGHI y
  buffer 2, ≈ 70 ms.
- **Peso del runtime**: `ort-wasm-simd-threaded.wasm` pesa 14 MB (3.7 MB gzip); el browser lo cachea.
- **Un solo thread**: el wasm multi-thread exige cross-origin isolation (headers COOP/COEP). Para batch 1 no hace
  falta. WebGPU tampoco conviene: con un MLP de batch 1 el overhead de dispatch supera al cómputo.
- **Crear una sesión bloquea el worker** unos cientos de ms (se tapa con el pre-buffer descrito arriba).
- La frecuencia de muestreo del `AudioContext` se fija a la del modelo (22050 Hz); el browser remuestrea a la placa.
