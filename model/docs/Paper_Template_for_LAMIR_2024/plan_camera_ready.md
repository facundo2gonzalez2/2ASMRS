# Plan de mejoras: versión camera-ready LAMIR 2026

> **Estado (2026-10-29):** plan aplicado en `paper.tex`, salvo lo que queda fuera por decisión del autor: el
> baseline de cross-fade de espectrogramas (omitido) y los opcionales de la §7 (para después, si hay tiempo).
> Fig. de transición cruzada regenerada en PDF vectorial con `model/scripts/plot_cross_transition.py`. Tabla 1
> generada con `model/scripts/summarize_interpolation_metrics.py`. Se agregó la UMAP (`figs/umap_notitle.png`,
> el mismo PNG sin el título en castellano). Resultado: 4 páginas técnicas + ethics + referencias.
> Pendientes del autor: afiliaciones y e-mails, hardware de la medición de tiempo real (`% TODO` en §4.5),
> licencias de los audios de piano, y verificar que los links de Drive sean públicos.

Paper: `paper.tex` (versión enviada, commit `884bafc`). Resultado: **Accept**, con la condición de que en la
camera-ready se resuelvan las limitaciones de evaluación y las afirmaciones sin respaldo.

Restricción de este plan: **no agregar experimentos nuevos ni regenerar audios**. Todo lo que figura abajo sale de
(a) reescribir texto, (b) material que ya existe en la tesis o en el repo, o (c) números que se pueden leer
de CSVs y logs que ya están calculados. Los pocos ítems que implicarían correr algo están separados en la §7
como opcionales.

---

## 1. Diagnóstico general

Los tres revisores coinciden en que la idea es novedosa y está bien explicada. Las críticas se agrupan en
cuatro ejes. Abajo cada uno con su prioridad para el accept:

| Eje | Quién | Gravedad | Se resuelve sin experimentos |
|---|---|---|---|
| **A. Afirmaciones que van más allá de la evidencia** (contribuciones "validated", "decisive", FAD "robust") | R1, meta | Alta | **Sí**: reescribir |
| **B. Métrica poco documentada / poco justificada** (tamaños, FAD por clip, valores "sospechosamente" chicos, motivación) | R1, R2 | Alta | **Sí**: documentar + calibración que ya está en la tesis |
| **C. Falta de medidas objetivas / baseline / comparación** | R1, R2, R3 | Media | **En parte**: tabla de indicadores a partir de CSVs existentes + baseline de cross-fade ya calculado |
| **D. Tiempo real afirmado pero no probado** | R2, R3 | Media | **Sí**: el sintetizador web ya mide costo por frame y latencia |
| E. Reproducibilidad (código, audios, piano) | R1, R2, R3 | Media | **Sí**: la camera-ready deja de ser anónima → links |
| F. Forma: figura de baja calidad, páginas, frase larga, Fig. 1 confusa | R1, R2 | Baja | **Sí** (la figura requiere re-plotear desde CSV, sin recalcular) |

Lo que **no** se puede resolver sin experimentos, y conviene reconocerlo explícitamente como limitación o trabajo
futuro: test de escucha formal (R1, R3), comparación contra sistemas existentes como NSynth o SoundMorpher (R2),
FAD de reconstrucción (R1.1), otros embeddings (R1.6) y KAD (R1.5).

---

## 2. Inconsistencias encontradas al cotejar el paper con el código

Conviene corregirlas sí o sí; un revisor experto (R1) podría encontrarlas.

1. **Arquitectura incompleta (§3.4).** El paper dice `1025→(512,256,128,64,32,16,8)→4`. Los `hparams.yaml` de
   los modelos de inferencia dicen `encoder_layers: 1025,1024,512,…,8,4` y `decoder_layers: 8,…,1024,1025`, es
   decir que **falta la capa de 1024**. El modelo completo tiene **3.51 M** parámetros (`parameters: 3512647`).
   El decoder solo tiene ~1.76 M (≈7 MB en float32, el tamaño de cada `web/public/models/weights/*.bin`).
   *Ojo:* revisar si los sweeps de dimensión latente (`experiment_latent_dim_*`) usaron la misma arquitectura
   antes de escribir una sola.
2. **Qué se interpola (R2, l.180).** `interpolar_vae` (`experiments/interpolate.py:96`) interpola **todo el
   `state_dict`** (encoder y decoder), **tensor por tensor**: cada tensor se aplana y se hace SLERP con su propio
   ángulo θ. El paper da a entender un único vector de pesos ("θ is the angle between w_A and w_B"). Para generar
   solo se usa el decoder interpolado; en el morphing audio-a-audio, cada audio se codifica con el encoder de su
   modelo puro.
3. **BatchNorm (R2, l.186).** `running_mean` y `running_var` **son float y sí se interpolan**. Lo único que no se
   interpola es `num_batches_tracked` (entero, contador de pasos de entrenamiento). Ese valor **difiere** entre
   modelos (cada fine-tuning corre una cantidad distinta de pasos) y **no se usa en modo eval**. El paper dice
   "Batch Normalization counters", que es correcto pero ambiguo; R2 lo leyó como si fueran las estadísticas.
   El umbral de fallback a LERP también conviene explicitarlo: |cos θ| > 0.9995.
4. **Qué compara la FAD (R1.5, R1.6).** `audio_comparator.get_audio_similarity_fad` calcula una distancia de
   Fréchet **entre dos clips individuales**, usando como "conjunto" los frames de embedding de MERT de cada clip:
   - Clip de 48 frames STFT (≈1.11 s) en los experimentos de interpolación. MERT-v1 genera 75 frames/s a 24 kHz,
     así que son ≈83 vectores de 768 dimensiones por clip.
   - Clip de 64 frames (≈1.49 s) en la calibración y en el baseline de cross-fade, ≈111 vectores.
   - Con menos vectores que dimensiones, **las covarianzas son singulares**. El código las regulariza sumando
     ε·I, con ε entre 1e-6 y 1e-1 hasta que `sqrtm` converge.
   - Similitud: `sim = exp(-FAD/k)` con **k = 585**. El paper no da el valor de k.

   Esto no es la FAD "de dataset" estándar, y R1 tiene razón en que es inestable con pocas muestras. Hay que
   **decirlo explícitamente** y llamarla, por ejemplo, *clip-level Fréchet distance on MERT frame embeddings*.
5. **Referencias de la Fig. 1.** En el experimento de interpolación cruzada (`interpolate.py`, config en `main()`)
   el Z es **gaussiano aleatorio y constante** en los 48 frames, con **fase aleatoria**, sobre 25 muestras. Las
   referencias A/B/testigo son la salida de cada **modelo puro** con ese mismo Z, no audio original. Por eso:
   - la similitud en α=0 con la fuente es ≈0.91 y no 1: el render con fase aleatoria cambia entre corridas, así
     que ese ≈0.91 funciona como techo práctico;
   - el contenido es estático (Z constante), lo que explica parte de la "baja varianza" que señala R1.
6. **El rango útil de la métrica es estrecho y hay que explicarlo (R1.6).** En la calibración, dos audios
   originales de instrumentos distintos dan sim ≈ 0.82, que equivale a FAD ≈ 116. Una sim de 0.91 equivale a
   FAD ≈ 55. Es decir, **los FAD crudos no son chicos**: lo que los comprime es la transformación exp(−FAD/585).
   Conviene reportar también FAD crudo, o al menos el piso y el techo.
7. **Fig. 1: el caption no coincide del todo con lo que muestra.** Dice que los modelos *Scratch* "barely change
   until α approaches 1". En Scratch-β=0.001 la similitud al piano baja gradualmente desde α=0; lo que se demora
   es la subida hacia la guitarra. Ajustar el texto.
8. **La afirmación "regularization is the more impactful factor" no es consistente en todos los experimentos.**
   En la interpolación cruzada (ver tabla de la §4), **la inicialización común es la que más mejora la
   monotonía** hacia el destino (ρ ≈ 0.9 contra ≈ 0.4–0.5 en Scratch), mientras que **β es lo que estabiliza al
   testigo** (rango 0.03 contra 0.10) y centra el cruce cerca de α=0.5. La afirmación de "β más impactante" sale
   solo del experimento agregado. Recomiendo decir que **ambos factores contribuyen a aspectos distintos**, en vez
   de jerarquizarlos.
9. **"Future work: real-time inference and weight interpolation"** ya está hecho: el sintetizador en `web/`
   interpola pesos por frame en el browser (ver §5).

---

## 3. Material existente que se puede reutilizar

| Material | Dónde está | Qué responde |
|---|---|---|
| **Calibración de la FAD** con cross-fade de formas de onda originales (2 pares, 20 pares de fragmentos de 1.5 s, testigo). Sale monótona, se cruza en α≈0.5, testigo plano y piso de ≈0.82 | Tesis §"Elección de métricas" (líneas ~543–553), `model/crossfade_fad_calibration_output/` (CSV + PNG), `experiments/crossfade_fad_calibration.py` | R2 (métrica poco justificada, l.245), R1.6 (valores chicos o colapso), R1 contribución iii |
| **Baseline de cross-fade de espectrogramas vs. morphing VAE**, 20 pares × 2 pares de instrumentos, Checkpoint-β | `model/crossfade_fad_output_checkpoint_beta_0.001/` (CSV + 2 PNG) | R2 (sin baseline). Ver la advertencia en §6 |
| **CSV del experimento de interpolación cruzada**, 4 configuraciones × 2 pares × 25 muestras × 11 α, con cosine y FAD | `model/interpolation_random_25_samples_ckpt_vs_scratch_gaussian_latest2/` | R1.2/R1.4: indicadores objetivos (tabla en §4) |
| **MSE de validación de los 18 modelos** | Logs TensorBoard en `model/inference_models/*/*/version_0/` (ya extraídos abajo) | R1.1 (tabla de reconstrucción), R2 l.36 |
| **Sweep de dimensión latente con modelo base** (por instrumento contra base, con el mismo tamaño) | Tesis fig. `1_comparacion_por_instrumento.png` | R2 l.36: el modelo compartido reconstruye peor |
| **UMAP** del espacio latente del modelo base | `figs/umap.png`, tesis §"Análisis del espacio latente" | R1.3 (visualizaciones del espacio latente) |
| **Costo y latencia en tiempo real** del sintetizador web (ONNX Runtime Web, wasm, 1 thread) | `web/README.md`, `npm test` | R2, R3 (tiempo real) |
| **Audios de ejemplo**: voz→piano, guitarra→piano, piano→guitarra Scratch-β y Scratch-AE (contraejemplos), morphing contra cross-fade | Links de Drive en tesis §"Ejemplos de audio" | R1 ("audio examples must be linked"), R3 |
| **Lista de archivos de piano**: 20 archivos de chosic.com y pixabay | `model/data_instruments/piano/` | R2 l.110 |

---

## 4. Números ya calculados para la tabla objetiva

### 4.1 Reconstrucción (MSE de validación, mejor época; logs TensorBoard existentes)

| Configuración | bass | guitar | piano | voice | **media** |
|---|---|---|---|---|---|
| Checkpoint-β-VAE | 0.00310 | 0.00320 | 0.00284 | 0.00422 | **0.00334** |
| Checkpoint-AE | 0.00203 | 0.00184 | 0.00164 | 0.00280 | **0.00208** |
| Scratch-β-VAE | 0.00310 | 0.00406 | 0.00298 | 0.00415 | **0.00357** |
| Scratch-AE | 0.00200 | 0.00183 | 0.00158 | 0.00273 | **0.00204** |
| Base model (4 instr.) β=0.001 / β=0 | — | — | — | — | **0.00361 / 0.00228** |

- Los modelos especializados reconstruyen algo mejor que el modelo base del mismo tamaño: 0.00206 contra 0.00228
  con β=0 (≈10% menos) y 0.00334 contra 0.00361 con β=0.001. En el sweep de dimensión latente sobre el dataset
  chico, la diferencia a dim=4 es mayor, ≈20–27% (leer los valores exactos de la figura de la tesis). Esto da
  **evidencia propia** para la frase de la l.36 (R2), siempre que se formule con moderación.
- Caveat a mencionar: cada modelo tiene su propio split de validación del 5%, así que la comparación es
  indicativa y no pareada.

### 4.2 Interpolación cruzada (Fig. 1): indicadores por muestra (media ± std sobre 25 Z), sacados del CSV existente

ρ es la correlación de Spearman entre α y la similitud, calculada sobre los 11 valores de α de cada muestra.
"Rango testigo" es max−min de la similitud al testigo a lo largo de α. "Cruce" es el α donde se cruzan las curvas
medias de fuente y destino.

| Config | par | ρ fuente | ρ destino | rango testigo | cruce α |
|---|---|---|---|---|---|
| **Ckpt-β-VAE** | piano→guitar | −0.88±0.17 | +0.89±0.13 | **0.033±0.020** | **0.55** |
| | piano→voice | −0.85±0.20 | +0.84±0.18 | **0.038±0.013** | **0.59** |
| Ckpt-AE | piano→guitar | −0.81±0.14 | +0.95±0.05 | 0.103±0.022 | 0.28 |
| | piano→voice | −0.84±0.17 | +0.90±0.18 | 0.084±0.010 | 0.26 |
| Scratch-β-VAE | piano→guitar | −0.67±0.23 | +0.50±0.36 | 0.036±0.015 | 0.48 |
| | piano→voice | −0.74±0.15 | +0.49±0.36 | 0.045±0.023 | 0.91 |
| Scratch-AE | piano→guitar | −0.79±0.08 | +0.36±0.26 | 0.041±0.017 | 0.76 |
| | piano→voice | −0.58±0.12 | +0.42±0.32 | 0.045±0.015 | 0.40* |

\* En Scratch-AE piano→voice el "cruce" se produce porque ambas curvas caen, no porque haya transferencia.

Lectura honesta, que es la que conviene poner en el paper:
- **La inicialización común** es la que hace que el acercamiento al destino sea monótono: ρ_dest ≈ 0.84–0.95
  contra 0.36–0.50 en Scratch.
- **β** es lo que mantiene estable al testigo dentro de Checkpoint (0.03–0.04 contra 0.08–0.10) y lleva el cruce
  cerca de α=0.5, cuando en Checkpoint-AE está en ≈0.27.
- Solo **Checkpoint-β-VAE** cumple las tres condiciones a la vez. Esto respalda la conclusión central con números
  y obliga a suavizar la afirmación "β es el factor más impactante" (ver §2.8).

El script de una pasada que generó estas tablas lee `interpolation_raw_metrics.csv` y los logs TensorBoard. Si se
decide incluirlas, conviene guardarlo como `model/scripts/summarize_interpolation_metrics.py` para que sean
reproducibles.

### 4.3 Calibración y baseline de cross-fade (similitud media; CSVs existentes)

| | α=0 | α=0.2 | α=0.5 | α=0.8 | α=1 |
|---|---|---|---|---|---|
| Calibración (onda original) piano→guitar: fuente / destino / testigo | 1.00/0.82/0.82 | 0.94/0.85/0.83 | 0.90/0.89/0.83 | 0.85/0.95/0.82 | 0.82/1.00/0.82 |
| Cross-fade de espectrogramas piano→guitar: fuente / destino | 1.00/0.83 | 0.94/0.84 | 0.91/0.86 | 0.88/0.90 | 0.83/1.00 |
| Morphing VAE (Ckpt-β) piano→guitar: fuente / destino | 1.00/0.83 | 0.88/0.83 | 0.84/0.84 | 0.83/0.86 | 0.83/1.00 |

---

## 5. Evidencia de tiempo real (sin medir nada nuevo)

Del sintetizador web (`web/README.md`, medido con `npm test`):
- Decoder: ~1.76 M parámetros, ≈3.5 MFLOP por frame. El decoder interpolado evalúa A·x y B·x, así que son ≈7 MFLOP.
- **Costo por frame del decoder interpolado en el browser: ~1–2 ms** (ONNX Runtime Web, wasm, 1 thread). Los
  ángulos SLERP se precalculan como constantes al crear la sesión (200–600 ms); por frame solo se computa lo que
  depende de α.
- Reconstrucción de fase por frame: aleatoria 0.2 ms, RTPGHI 0.6 ms, RTISI-LA (lookahead 2, 8 it) ~5 ms.
- **Presupuesto por frame: hop de 512/22050 = 23.2 ms.** Factor de tiempo real ≈ 0.1–0.3 según el método de fase.
- Latencia total: ≈70 ms (RTPGHI, buffer 2) a ≈140 ms (RTISI-LA, defaults).

**Pendiente:** anotar el hardware (CPU y browser) en el que se midieron esos números; un revisor lo va a pedir.

Con esto se puede **mantener** el claim de tiempo real como resultado (uno de los dos caminos que ofrece R2) y
sacar "real-time inference" de trabajo futuro. Si se prefiere no depender del web synth, el otro camino que ofrece
R2 es bajar el claim a "lightweight, trainable on modest hardware" y dejar el tiempo real como trabajo futuro.
**Recomiendo la primera opción**: es la evidencia más fuerte que tenemos y además responde a R3.

---

## 6. Respuesta punto por punto → acción concreta

### Reviewer 1

| # | Comentario | Acción | Sección |
|---|---|---|---|
| G1 | Citar Murata et al. (model merging en TTS) y citas "obligatorias" de la técnica | Agregar a Related Work un párrafo corto sobre **weight-space interpolation / model merging**: Murata 2024, Model Soups (Wortsman 2022), Task Arithmetic (Ilharco 2023), y para explicar *por qué* importa la init común: Linear Mode Connectivity (Frankle 2020) y Neyshabur 2020 ("same basin"). Para SLERP: Shoemake 1985 y White 2016. Bib verificada en §9 | §2 |
| G2.1 | "validated" es demasiado | Cambiar a "propose and evaluate" o "provide initial evidence" | §1, §5 |
| G2.2 | "decisive" requiere ablaciones | Reformular: el diseño **2×2 (init × β) es una ablación factorial**, y Scratch-AE es el baseline sin ninguno de los dos factores. Decirlo así y respaldarlo con la tabla §4.2. Cambiar "decisive" por "both contribute, to different aspects" | §1, §3.5, §4 |
| G2.3 | Eliminar la contribución (iii) sobre FAD | **Eliminarla.** La comparación con cosine queda como justificación metodológica (1–2 frases), no como contribución. Sacar también "FAD … proved robust" de las conclusiones y "on which the community has converged" de Related Work | §1, §2, §4.3, §5 |
| E1 | Tablas FAD de reconstrucción | Sin experimentos nuevos: agregar la **tabla de MSE de reconstrucción** (§4.1) y reportar piso y techo de la métrica (≈0.82 entre instrumentos distintos, ≈0.91 de techo con fase aleatoria). La FAD de reconstrucción propiamente dicha queda como opcional en §7 | §4.1 |
| E2 | §4.1 depende de escucha informal | Quitar "informal listening confirmed…" o pasarlo a una nota marginal. Reemplazarlo por los valores numéricos del codo (MSE y KL por β; están en la Fig. Pareto de la tesis) | §4.1 |
| E3 | §4.2 necesita visualización; los audios deben estar linkeados | UMAP: o una mini-figura, o (si no hay espacio) la página complementaria. Audios: **link en la camera-ready**, que ya no es anónima (GitHub Pages o Drive) + link a la demo web | §4.2, §4.5 |
| E4 | Afirmaciones de §4.3 sin respaldo | Reemplazar §4.3 por "**Metric sanity check**": la calibración con cross-fade de onda original (resultado conocido de antemano, sin modelo). La comparación con cosine se reduce a una frase, sin afirmaciones de superioridad | §4.3 |
| E5 | Tamaño de los conjuntos comparados por FAD; mirar KAD | Documentar: FAD **por clip** sobre ≈83 frames MERT (768-d) por clip, regularización ε·I, k=585, 25 muestras por punto α, 11 valores de α, 2 pares de instrumentos. Reconocer la inestabilidad con pocas muestras y **citar KAD** como trabajo futuro | §3.7, §5 |
| E6 | Valores FAD "sospechosamente chicos" y con poca varianza; ¿otros embeddings? | Explicar que se grafica la similitud exp(−FAD/k), no FAD crudo (0.82 ↔ FAD≈116). Usar la calibración como evidencia de que la métrica no está colapsada (sube y baja monótonamente, testigo plano, piso estable). Explicar que la baja varianza viene en parte de Z constante + fase aleatoria. Otros embeddings → limitación / trabajo futuro | §3.7, §4.3, §4.4 |
| E7 | Excede 4 páginas | La versión enviada tiene 4 páginas de contenido técnico + ethics + referencias, que es compatible con la regla 4+N. Igual hay que **ganar espacio** para lo nuevo (presupuesto en §8) | — |
| E8 | Figura de baja calidad | Re-exportar en **PDF vectorial** desde el CSV (modo `plot_only`, sin recalcular). Ver §8 | Fig. 1 |
| Q12 | Reproducibilidad: "Strongly disagree" | Camera-ready no anónima → link al **repo** (`github.com/facundo2gonzalez2/2ASMRS`; verificar que sea público), a los modelos entrenados (`web/public/models/` ya está commiteado) y a la demo. Agregar los hiperparámetros que faltan (arquitectura corregida, k, ε, tamaños de clip, seeds) | §3, §4 |

### Reviewer 2

| Línea / tema | Acción |
|---|---|
| Métrica poco sólida, sin motivación en la literatura | (a) Citar los antecedentes que usan FAD para morphing o timbre (SoundMorpher y Chen et al. evalúan con FAD; verificar en sus papers antes de afirmarlo) y Gui et al. para FAD con embeddings musicales. (b) La **calibración** de §4.3 justifica empíricamente la métrica *independientemente* del sistema evaluado, que es justamente su preocupación ("selecting both the metric and the system at the same time is risky"). (c) Declarar sus límites |
| Sin comparación con otros sistemas | Ser explícito: comparar con NSynth o modelos de difusión no es directo (otra escala y otro tipo de control), y se deja como trabajo futuro. Como baseline **training-free**, mencionar el cross-fade, que ya está calculado (ver advertencia abajo). Mencionar como baseline pendiente la interpolación latente en el modelo base compartido, que es la alternativa que el paper dice superar |
| Tiempo real no probado | §5: agregar 3–4 líneas con costo por frame, presupuesto y latencia del sintetizador web. Mover "real-time" de trabajo futuro a resultado |
| l.36: afirmación clave sin respaldo | Respaldar con datos propios (§4.1: los especializados reconstruyen 7–10% mejor, y ≈20–27% en el sweep) y suavizar a "tends to"; o citar trabajo que lo muestre si se encuentra alguno verificable |
| l.110: fuente del piano | Especificar: "20 royalty-free/public-domain classical piano recordings from Chosic and Pixabay (list in the repository)". **Verificar licencias** antes de escribir "public domain" |
| l.180: ¿se interpola solo el decoder? | Aclarar (§2.2): se interpolan todos los parámetros tensor por tensor; para la síntesis solo se usa el decoder resultante |
| l.186: contadores de BatchNorm | Aclarar (§2.3): running mean/var se interpolan; solo `num_batches_tracked` (entero, distinto entre modelos, sin uso en inferencia) toma el valor del modelo más cercano |
| l.245: "trajectory through latent space" | Aclarar que **no** es interpolación latente entre instrumentos: es la secuencia de Z que se obtiene codificando un clip real de validación (cambia en el tiempo) y que se pasa por el decoder interpolado en pesos. Renombrar a "time-varying latent sequence from an encoded clip" |
| Fig. 1: arriba y abajo no se distinguen | Rotular filas ("Checkpoint init" / "Scratch init") y columnas ("β-VAE, β=0.001" / "AE, β=0"), compartir el eje y entre paneles y colorear de forma consistente fuente, destino y testigo |
| l.68: frase muy larga | Partir la oración de NSynth (l.68–73) en dos |

### Reviewer 3

| Tema | Acción |
|---|---|
| Solo 4 categorías y sin estudio perceptual | Agregar un párrafo de **Limitations** explícito: 4 fuentes, FAD como métrica principal, sin test de escucha formal. Los audios se linkean y queda prometido un listening test (MUSHRA o similar) para la versión extendida |
| Costo computacional y tiempo real | §5 |
| Ablaciones y evaluación subjetiva para una versión extendida | Mencionar en trabajo futuro (no requiere acción ahora) |

### Advertencia sobre el baseline de cross-fade (decisión tuya)

Los números de §4.3 muestran que, en α intermedios, el morphing VAE queda **a distancia similar de A y de B y cerca
del piso entre instrumentos** (≈0.84 a ambos en α=0.5). El cross-fade de espectrogramas, en cambio, queda alto
respecto de ambos (0.91 y 0.86), porque literalmente *contiene* las dos fuentes superpuestas. Hay dos lecturas:
- **Favorable:** la similitud FAD con los extremos mide la *presencia* de cada fuente. Un cross-fade la maximiza
  por construcción sin crear un timbre nuevo, mientras que el morphing produce un sonido que no es reducible a A
  ni a B, que es el objetivo de la fusión tímbrica (la misma distinción que hace Slaney entre mezclar y fusionar).
- **Desfavorable:** un revisor podría leerlo como que el sonido intermedio se degrada.

Recomendación: **incluir la calibración seguro** (es limpia y responde directamente a R2). El baseline de cross-fade
conviene incluirlo **solo como 1–2 frases** con la interpretación de arriba y la salvedad de que FAD no distingue
superposición de fusión, lo que motiva el test de escucha como trabajo futuro. No conviene presentarlo como
"VAE beats baseline", porque los números no lo dicen.

---

## 7. Opcionales de bajo costo (requieren correr algo; vos decidís)

Ordenados por relación costo/beneficio. Ninguno entrena modelos ni genera audios nuevos con los modelos, salvo el
primero.
1. **FAD de reconstrucción** (R1.1): para N≈20 clips de validación por instrumento, calcular la FAD entre el
   original y la reconstrucción de cada configuración. Son minutos de MERT. Es lo que R1 pidió "at the very least".
2. **Baseline intra-instrumento:** la similitud entre dos clips *distintos* del *mismo* instrumento. Hoy no está
   calculada y daría el techo realista de la métrica, más informativo que el auto-similar = 1.
3. **KAD sobre los WAVs ya generados** (todos están en los directorios de salida): recalcular la métrica sin
   regenerar audio, usando `kadtk`. Sirve para mostrar que el orden entre configuraciones se mantiene.
4. **Otro embedding** (CLAP o VGGish) sobre los mismos WAVs, para responder a R1.6.

Si no se hace ninguno, todos van a "Limitations / Future work" con una frase.

---

## 8. Presupuesto de espacio (la camera-ready sigue en 4+N)

**Verificar** en las instrucciones de camera-ready de LAMIR 2026 si cambia el límite de páginas o si se permite
una página extra.

Agregados (estimación en líneas de columna):
- Párrafo de related work sobre model merging: +7
- Aclaraciones del método (per-tensor, BN, ε, FAD por clip, k, tamaños): +8
- Tabla compacta (§4.1 medias + §4.2, solo Ckpt/Scratch × β; ~8 filas): +14
- Calibración de la métrica (texto, sin figura): +6
- Tiempo real: +5
- Limitaciones: +5
- Links (repo, audios, demo): +3
- **Total ≈ +48 líneas (≈ 0.8 columnas)**

Recortes:
- Fig. 1 de 0.88\textwidth a ~0.75\textwidth, o a 1 columna con 2×2 compacto y eje y compartido: −10 a −20
- Bajar §3.3 (VAE) a lo esencial; la ELBO y el reparameterization trick son de manual: −10
- §4.3 (cosine contra FAD) se reemplaza por la calibración (neto ≈ 0): −6
- Quitar "Audio Examples" como subsección y dejar una frase con link: −4
- Comprimir el primer párrafo de la introducción y el de diseño: −6
- Contribuciones más cortas (sin la iii): −3
- `\linenumbers` se quita en la camera-ready (no gana líneas, pero es obligatorio)
- **Total ≈ −40 a −50 líneas.** Ajustado pero viable. Si no entra, la tabla §4.2 puede reducirse a un solo par
  (piano→guitar, que es el de la Fig. 1).

### Figura (R1.8, R2)
- El script de la Fig. 1 está en `experiments/interpolate.py` (con `plot_only=True` lee el CSV
  `interpolation_raw_metrics.csv` y no recalcula nada). Cambios:
  `savefig(...pdf)`, `sharey=True`, rótulos de fila y columna, leyenda única, tamaño de fuente legible a 1 columna.
  Ojo: el módulo importa `audio_comparator`, que carga MERT al importar (tarda, pero no recalcula métricas).
- En LaTeX: `\includegraphics[width=\columnwidth]{figs/cross_transition.pdf}`.

---

## 9. Bibliografía nueva (verificada que existe; revisar campos y DOI antes de usar)

Recordatorio: en LAMIR, una referencia alucinada implica rechazo automático. Todas estas se verificaron por
búsqueda, pero **confirmar páginas y DOI en la fuente oficial**.

```bibtex
@inproceedings{murata2024attribute,
  author    = {Murata, Masato and Miyazaki, Koichi and Koriyama, Tomoki},
  title     = {An Attribute Interpolation Method in Speech Synthesis by Model Merging},
  booktitle = {Proc. Interspeech},
  year      = {2024},
  note      = {arXiv:2407.00766}
}
@inproceedings{wortsman2022soups,
  author    = {Wortsman, Mitchell and Ilharco, Gabriel and Gadre, Samir Yitzhak and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Morcos, Ari S. and Namkoong, Hongseok and Farhadi, Ali and Carmon, Yair and Kornblith, Simon and Schmidt, Ludwig},
  title     = {Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time},
  booktitle = {Proc. 39th Int. Conf. on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {162},
  pages     = {23965--23998},
  year      = {2022}
}
@inproceedings{ilharco2023task,
  author    = {Ilharco, Gabriel and Ribeiro, Marco Tulio and Wortsman, Mitchell and Gururangan, Suchin and Schmidt, Ludwig and Hajishirzi, Hannaneh and Farhadi, Ali},
  title     = {Editing Models with Task Arithmetic},
  booktitle = {Proc. Int. Conf. on Learning Representations (ICLR)},
  year      = {2023}
}
@inproceedings{frankle2020lmc,
  author    = {Frankle, Jonathan and Dziugaite, Gintare Karolina and Roy, Daniel M. and Carbin, Michael},
  title     = {Linear Mode Connectivity and the Lottery Ticket Hypothesis},
  booktitle = {Proc. 37th Int. Conf. on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {119},
  pages     = {3259--3269},
  year      = {2020}
}
@inproceedings{neyshabur2020transfer,
  author    = {Neyshabur, Behnam and Sedghi, Hanie and Zhang, Chiyuan},
  title     = {What is being transferred in transfer learning?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  year      = {2020}
}
@misc{white2016sampling,
  author = {White, Tom},
  title  = {Sampling Generative Networks},
  year   = {2016},
  note   = {arXiv:1609.04468}
}
@inproceedings{shoemake1985slerp,
  author    = {Shoemake, Ken},
  title     = {Animating Rotation with Quaternion Curves},
  booktitle = {Proc. 12th Annual Conf. on Computer Graphics and Interactive Techniques (SIGGRAPH)},
  pages     = {245--254},
  year      = {1985}
}
@misc{chung2025kad,
  author = {Chung, Yoonjin and Eu, Pilsun and Lee, Junwon and Choi, Keunwoo and Nam, Juhan and Chon, Ben Sangbae},
  title  = {{KAD}: No More {FAD}! An Effective and Efficient Evaluation Metric for Audio Generation},
  year   = {2025},
  note   = {arXiv:2502.15602}
}
```

Párrafo sugerido para Related Work (borrador):

> Interpolating the *weights* of networks has been studied mainly outside audio. Weight averaging of models
> fine-tuned from a shared pre-trained checkpoint improves accuracy without extra inference cost
> \cite{wortsman2022soups}, and weight differences act as composable "task vectors" \cite{ilharco2023task}. Such
> averaging works because models trained from a common starting point remain in the same loss basin and are
> linearly connected \cite{frankle2020lmc,neyshabur2020transfer}, which motivates our *Checkpoint* strategy.
> Closest to our setting, Murata *et al.* \cite{murata2024attribute} merge text-to-speech models to interpolate
> speaker and emotion attributes. Spherical interpolation \cite{shoemake1985slerp} is common for latent vectors of
> generative models \cite{white2016sampling}; we apply it per parameter tensor.

Esto además le da a la estrategia *Checkpoint* una **explicación teórica** con citas, que hoy falta.

---

## 10. Borradores de texto para las partes críticas

### Contribuciones (reemplazo de §1, l.98–106)

> **Contributions.** (i) We propose a timbre-morphing technique that interpolates the *weights* of per-instrument
> specialized VAEs, rather than the latent code of a single shared model, and evaluate it on four sound sources.
> (ii) Through a 2×2 ablation over initialization (shared checkpoint vs. scratch) and latent regularization
> (β-VAE vs. AE), we provide initial evidence that both factors contribute to well-behaved interpolations: a
> shared initialization yields monotonic transfer towards the target, while regularization keeps unrelated
> timbres unaffected and centres the transition. (iii) We show that the weight-interpolated decoder runs in real
> time in a web browser, and release code, models, audio examples and an interactive demo.

### Métrica (agregar a §3.7)

> FAD is computed between the distributions of frame-level MERT embeddings (768-d, 75 frames/s) of *two individual
> clips* of ≈1.1 s (≈83 frames each). Because the number of frames is smaller than the embedding dimension, the
> covariances are rank-deficient and are regularized with εI (ε ≤ 10⁻¹). We report sim = exp(−FAD/k) with k = 585.
> Each point averages 25 latent samples. This clip-level estimate is known to be biased for small sample sizes
> \cite{chung2025kad}; we therefore first verify its behaviour on a controlled case (Sec. 4.3).

### Sanity check de la métrica (reemplazo de §4.3)

> Before evaluating the models, we check the metric on a case whose outcome is known in advance and that involves no
> model: a waveform cross-fade (1−α)A + αB between original clips of two instruments (piano→guitar and
> voice→piano, 20 random 1.5 s clip pairs each). Similarity to A decreases monotonically from 1 to ≈0.82, similarity
> to B mirrors it, the curves cross at α≈0.5, and similarity to a witness instrument stays flat (≈0.82, std ≤ 0.02).
> Two clips of different instruments thus score ≈0.82 (FAD ≈ 116), which sets the useful range of the metric in
> our setting. Cosine similarity between mean MERT embeddings follows the same trend but with higher dispersion along
> time-varying latent inputs, so we use FAD in what follows.

### Tiempo real (nuevo, al final de §4)

> **Real-time operation.** The decoder has 1.76 M parameters (≈3.5 MFLOP per frame; ≈7 MFLOP for the interpolated
> decoder, which evaluates both endpoints). In a browser implementation (ONNX Runtime Web, single-threaded WASM,
> [CPU]), the interpolated decoder takes ≈1–2 ms per frame and real-time phase reconstruction (RTPGHI) 0.6 ms,
> against a 23.2 ms hop budget; α and z can be modulated continuously with ≈70 ms end-to-end latency.

### Limitaciones (nuevo, en Conclusiones)

> **Limitations.** The evaluation covers four sources and relies on a clip-level FAD over a single embedding model;
> no formal listening test was conducted, and FAD to the endpoints cannot distinguish a superposition of two sources
> from a fused timbre. We did not compare against large latent-space morphing systems, whose scale and control
> paradigm differ. Future work includes a perceptual study, KAD \cite{chung2025kad} and alternative embeddings, a
> latent-interpolation baseline on the shared base model, and extended ablations.

---

## 11. Checklist final

- [ ] Des-anonimizar: autores reales, quitar "Retain these fake authors", `\authorname`, agradecimientos si se permiten
- [ ] Quitar `\usepackage{lineno}` y `\linenumbers`
- [ ] Corregir la arquitectura (capa de 1024) y agregar la cantidad de parámetros
- [ ] Aclarar SLERP por tensor, interpolación de todo el modelo, BN y umbral de fallback
- [ ] Documentar la FAD por clip: frames, dims, ε, k, N muestras, α, pares
- [ ] Reescribir contribuciones y eliminar la (iii)
- [ ] Reemplazar §4.3 por la calibración
- [ ] Tabla de reconstrucción e indicadores de interpolación (§4); guardar el script que la genera
- [ ] Suavizar "β more impactful" → "both contribute to different aspects"
- [ ] Sacar "informal listening" de §4.1 y dar los números del codo
- [ ] Párrafo de model merging + 8 citas nuevas (verificar DOI y páginas)
- [ ] Párrafo de tiempo real (anotar el hardware de la medición)
- [ ] Párrafo de limitaciones
- [ ] Fuente del piano especificada (y licencias verificadas)
- [ ] Aclarar l.245 ("time-varying latent sequence from an encoded clip")
- [ ] Partir la frase de la l.68
- [ ] Fig. 1 vectorial, con rótulos de fila y columna, eje y compartido y caption corregido
- [ ] Links: repo (verificar que sea público), audios, demo web
- [ ] Revisar consistencia de abstract y conclusiones con los nuevos claims ("validated" aparece también en §5)
- [ ] Compilar y verificar 4 páginas técnicas + N
