# Antecedentes en morphing de audio

Documento de trabajo para la sección *Antecedentes en morphing de audio* (Cap. Introducción). Reúne un repaso de las técnicas usadas históricamente para lograr *audio morphing*, agrupadas por familia de enfoque, con sus ventajas, desventajas y las diferencias concretas frente a la propuesta de esta tesis.

## Recordatorio: la propuesta de esta tesis

**Morphing tímbrico mediante interpolación de pesos de modelos VAE.** Entrenamos un β-VAE *fully-connected* pequeño (β = 0.001, espacio latente de dimensión 4) **especializado e independiente por instrumento** (piano, guitarra, bajo, voz) y luego generamos timbres híbridos **interpolando directamente en el espacio de PESOS** de esos modelos (LERP y SLERP, con un coeficiente α ∈ [0,1]). Trabajamos sobre espectrogramas de **magnitud** (STFT, *win* 2048 / *hop* 512, 22050 Hz), procesando cada *frame* de forma independiente (sin modelado temporal explícito) y reconstruyendo la fase con Griffin-Lim / PGHI / fase aleatoria. Los modelos son deliberadamente chicos y livianos, con la mira puesta en la ejecución en **tiempo real sobre hardware modesto**. Evaluamos con Fréchet Audio Distance (FAD) sobre *embeddings* de MERT, similitud coseno y UMAP.

**El eje que ordena todo este capítulo es *dónde ocurre la interpolación*.** La literatura interpola en la **señal** (DSP clásico) o en el **espacio latente / de ruido** de un modelo generativo; nuestra contribución es mover la interpolación al **espacio de parámetros (pesos)** de modelos especializados por instrumento.

---

## Tabla comparativa

| Antecedente | Año / Venue | Familia técnica | Qué morphea | Dónde interpola | ¿Estados intermedios controlables? | Tiempo real / tamaño |
|---|---|---|---|---|---|---|
| **Slaney, Covell & Lassiter** — *Automatic Audio Morphing* | 1996 / ICASSP | DSP clásico (análisis-síntesis), **sin aprendizaje** | Voz / habla genérica | En la **señal**: warping + cross-fade de espectrogramas descompuestos | Sí (parámetro λ), offline | No es objetivo; offline |
| **Engel et al. (NSynth)** — *WaveNet Autoencoders* | 2017 / ICML | Autoencoder neuronal (WaveNet autorregresivo) | Timbre entre instrumentos | En el **espacio latente** de un **único** modelo | Sí, por interpolación latente | No (autorregresivo, la demo pre-renderiza) |
| **Mancusi et al.** — *Latent Diffusion Bridges* | 2024 / ICASSP '25 | Difusión latente (DDIB / *Schrödinger bridges*) | **Timbre transfer** (no morphing) | En el **latente compartido** (prior gaussiano) vía ODE | No (transferencia total A→B) | No (ODE ~100 pasos, H100) |
| **Niu, Zhang & Martin** — *SoundMorpher* | 2024 / preprint IEEE | Difusión latente preentrenada (AudioLDM2) | Sonidos *open-world* (timbre, música, ambiente) | En el **ruido/latente** (SLERP) + *embeddings* de texto | Sí (static/dynamic/cyclostationary) | No (LoRA + búsqueda binaria por par) |
| **Chen et al.** — *Guitar Tone Morphing* | 2025 / APSIPA ASC | Difusión latente + LoRA **vs.** autoencoder (Music2Latent) | Tonos/efectos de **una misma guitarra** | En el **latente** (SLERP); variante *dual-LoRA* interpola pesos de la U-Net | Sí | Aspiración, no demostrado |
| **Chu et al.** — *Mix2Morph* | 2026 / preprint | Difusión latente texto-a-audio (transformer) | **SFX / sonidos generales** (*sound infusion*) | Dentro de **un** modelo, guiado por texto | No (morph estático asimétrico, sin α) | No (difusión, datos propietarios) |
| **Esta tesis** | 2026 | β-VAE *fully-connected* pequeño | Timbre entre instrumentos | En el **espacio de PESOS** de modelos independientes | Sí (α ∈ [0,1] continuo) | **Sí es el objetivo**; modelos chicos |

---

## 1. Enfoque clásico de DSP: análisis–síntesis sobre la señal

### Slaney, Covell & Lassiter (1996) — *Automatic Audio Morphing* (ICASSP-96)

**Enfoque.** Procesamiento de señales clásico, **sin ninguna forma de aprendizaje**: ni redes neuronales, ni espacio latente, ni entrenamiento. Es un *pipeline* de análisis–síntesis sobre espectrogramas de magnitud, apoyado en MFCC, *dynamic time warping* (DTW) y programación dinámica para el pitch. Se posiciona como sucesor del análisis sinusoidal: en lugar de trackear sinusoides y su fase, trabaja directamente sobre la magnitud y recupera la fase al final por inversión iterativa del espectrograma.

**Tipo de morphing.** Morphing de **sonido a sonido genérico**, inspirado explícitamente en el morphing de imágenes/video. Sus ejemplos son de voz y voz cantada (vocal /a/→/i/, palabras "morning"→"corner"). Representa cada sonido como un punto en un espacio perceptual de ejes ortogonales — **tiempo, forma espectral suave y pitch** — y el morph es un camino entre dos puntos de ese espacio.

**Método (representación → matching → interpolación → inversión).**
1. **Descompone** el espectrograma en un *smooth spectrogram* (forma espectral / formantes, vía MFCC) y un *pitch spectrogram* (residuo con la estructura armónica, obtenido por división). El truco central: un cross-fade de espectrogramas crudos suena a *dos* sonidos simultáneos (dos pitches); separando forma espectral de pitch se evita ese artefacto.
2. **Matching** automático: DTW alinea temporalmente ambos sonidos y la programación dinámica estima un contorno de pitch suave; luego estira/comprime el eje de frecuencia para alinear los armónicos.
3. **Interpolación** 1-D con cross-fade de las señales *warpeadas*: `s(λ,t) = (1−λ)·s₁(t₁) + λ·s₂(t₂)`.
4. **Inversión** iterativa del espectrograma de magnitud para recuperar fase y volver a la forma de onda.

**Ventajas.** Correspondencia automática (sin marcado manual); genérico (magnitud puede representar cualquier sonido); separación en dimensiones perceptuales independientes que se manipulan por separado; barato computacionalmente (no hay entrenamiento).

**Desventajas.** Requiere una etapa de **matching explícito** cuya función hay que elegir según la tarea; es un ensamble de heurísticas de DSP sensible a la calidad de la estimación de pitch/voicing; cuello de botella en la inversión de fase; probado solo sobre voz; no apunta a tiempo real.

**Diferencias con nuestra propuesta.**
- Interpola la **señal** (tras alinearla y descomponerla), no la **parametrización** de un modelo.
- No aprende nada: no hay red neuronal ni espacio latente.
- Necesita **matching previo** (DTW + alineamiento armónico); nuestra correspondencia queda implícita en los pesos aprendidos.
- El tiempo es un **eje explícito** que se alinea; nosotros procesamos cada *frame* de forma independiente, sin modelado temporal.
- Morphea voz genérica; nosotros, timbre entre instrumentos.

**Relevancia.** Es el trabajo **fundacional** del morphing de audio automático. Fija el vocabulario y la estructura conceptual (representación → matching → interpolación → inversión) que la literatura posterior retoma o busca superar, y sirve como línea de base "DSP clásica sin aprendizaje" frente a la cual contrastamos el enfoque neuronal.

---

## 2. Autoencoders neuronales: interpolación en el espacio latente

### Engel et al. (2017) — *Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders* / NSynth (ICML 2017, proyecto Magenta)

**Enfoque.** Un **WaveNet autoencoder**: un *encoder* convolucional temporal (30 capas de convoluciones dilatadas) produce un *embedding* temporal `Z`, y un *decoder* WaveNet **autorregresivo** (una muestra por vez) sintetiza el audio condicionado en ese *embedding*. Presentan también un *baseline* espectral (autoencoder convolucional sobre log-magnitud + Griffin-Lim). Notablemente, **probaron una variante variacional del latente y la descartaron** por poco efectiva (el decoder es tan potente que ignora la latente): el modelo principal usa un *embedding* **determinista**.

**Dataset NSynth.** ~306.000 notas musicales individuales de 1006 instrumentos, monofónicas de 4 s a 16 kHz, anotadas por *source* (acoustic/electronic/synthetic), *family* (11 familias) y *qualities*. Se volvió un banco de pruebas de referencia del área ("el MNIST/ImageNet del audio").

**Cómo hace morphing.** Interpola **en el espacio de *embeddings* Z de un único modelo** entrenado sobre todos los instrumentos: codifica dos instrumentos, interpola linealmente sus `Z` y decodifica el punto intermedio. La clave frente a mezclar audio: la mezcla en el dominio de la señal da una **superposición** (los dos suenan a la vez), mientras que interpolar en `Z` **fusiona** ambos timbres en un sonido híbrido nuevo, creando estructura armónica que no estaba en ninguno de los dos. El **"NSynth Sound Maker"** de Magenta materializa esta idea en una UI que ubica 4 instrumentos en las esquinas de una grilla x-y y mezcla los más cercanos.

**Ventajas.** Mejor reconstrucción y coherencia de fase que el *baseline* espectral; espacio latente semánticamente rico que *fusiona* timbres (no superpone); generaliza a duraciones/secuencias no vistas.

**Desventajas.** *Decoder* autorregresivo **muy lento**, no apto para tiempo real y con fuerte requerimiento de GPU (la propia demo tuvo que **pre-renderizar** todo el audio para ser jugable); contexto temporal limitado; distorsión por mu-law de 8 bits; modelo y dataset grandes.

**Diferencias con nuestra propuesta.**
- **Locus de la interpolación**: NSynth interpola en el **espacio latente** de **un único** modelo entrenado sobre todos los instrumentos; nosotros interpolamos los **pesos** de modelos **independientes** por instrumento. Esta es la diferencia conceptual central.
- Representación: *waveform* crudo autorregresivo vs. nuestro espectrograma de magnitud *frame a frame*.
- Latente: *embedding* determinista temporal (125×16) vs. nuestro β-VAE probabilístico de dimensión 4.
- Tamaño y tiempo real: red profunda que precomputa audio vs. VAE chico pensado para tiempo real.

**Relevancia.** Es el **antecedente más cercano en espíritu** a la tesis: estableció el paradigma de morphing tímbrico entre instrumentos por interpolación en un espacio latente neuronal, y su *Sound Maker* prefigura la misma UI de mezcla entre 4 instrumentos que perseguimos. Nos diferenciamos por mover la interpolación del **espacio latente** al **espacio de pesos** y por priorizar modelos livianos.

---

## 3. Modelos de difusión

Familia dominante en los trabajos recientes (2024–2026). Comparten el uso de modelos de difusión (por lo general **latentes y preentrenados**), pesados y costosos en inferencia, e interpolan/operan en el espacio latente o de ruido. Dentro de ella conviene distinguir *timbre transfer* (transferencia total) de *morphing* (estados intermedios).

### 3.1 Mancusi et al. (2024/2025) — *Latent Diffusion Bridges for Unsupervised Musical Audio Timbre Transfer* (ICASSP 2025)

**Enfoque.** *Dual diffusion bridges* — una solución al problema de *Schrödinger bridge* basada en **DDIB** (traducción imagen-a-imagen adaptada al audio). Es **difusión latente**: opera sobre los *embeddings* de EnCodec (128 canales), no sobre la señal, lo que evita necesitar un vocoder. Entrena **un modelo de difusión independiente por instrumento**, cada uno con un **prior gaussiano común**, de forma **no supervisada** y con datos **no pareados**.

**Timbre transfer, no morphing.** Es una distinción importante para el capítulo: transfiere el timbre de un instrumento fuente a uno destino **preservando la melodía/contenido musical** del audio de entrada. Es una **transferencia total A→B**, no un continuo de estados híbridos. Sí ofrece un control continuo — el nivel de ruido σ del prior — pero regula el **balance melodía-vs-timbre**, no un cruce entre los dos timbres.

**Método.** Inferencia en dos pasos encadenados sobre la *probability-flow ODE*: (1) *forward ODE* con el modelo fuente lleva el latente del audio al prior gaussiano compartido; (2) *reverse ODE* con el modelo destino reconstruye desde ese prior en el dominio del instrumento objetivo. La preservación del contenido viene de la *cycle consistency* heredada de DDIB (que extienden con garantías teóricas). Datos: CocoChorales, 4 instrumentos orquestales (violín, flauta, cello, fagot). Evalúan con FAD (*embeddings* EnCodec), un clasificador y métricas de preservación de melodía (Basic Pitch + DTW), más test perceptual.

**Ventajas.** No requiere datos pareados; **modular por instrumento** — agregar uno nuevo es entrenar un modelo más sin reentrenar el resto (filosofía análoga a la nuestra); no necesita vocoder ni guiado por gradiente; buena preservación del contenido musical.

**Desventajas.** Difusión costosa (ODE ~100 pasos, dos pasadas), **no apta para tiempo real**; modelos pesados (U-Net, entrenamiento en H100); **no produce estados intermedios controlables** entre timbres; problemas con diferencias de octava (requiere *pitch-shifting*).

**Diferencias con nuestra propuesta.**
- Objetivo: **timbre transfer** que preserva melodía vs. nuestro **morphing** que sintetiza timbres híbridos intermedios con α.
- Interpola/transporta en el **latente compartido** (prior gaussiano) vía ODE; nosotros en el **espacio de pesos**.
- Preserva el contenido del audio de entrada; nosotros sintetizamos timbre desde un vector latente, sin atarnos a un contenido dado.
- **Puntos de contacto**: ambos son no supervisados, entrenan **un modelo por instrumento** de forma independiente con la propiedad de agregar instrumentos sin reentrenar, y ambos usan FAD.

**Relevancia.** Es el antecedente más cercano en la línea "un modelo independiente por instrumento sin datos pareados", pero encarna el **paradigma opuesto**: transferencia total de alta fidelidad mediante difusión costosa, en lugar de síntesis liviana de timbres híbridos por interpolación de pesos.

### 3.2 Niu, Zhang & Martin (2024) — *SoundMorpher: Perceptually-Uniform Sound Morphing with Diffusion Model*

**Enfoque.** Morphing sobre un **modelo de difusión latente texto-a-audio preentrenado (AudioLDM2)**, usado *training-free* en el sentido de que no reentrena el modelo completo, aunque sí realiza dos optimizaciones ligeras **por cada par de sonidos**: inversión textual de los *embeddings* condicionales y adaptación con **LoRA**. Usa **DDIM** para invertir los audios a su ruido inicial e interpolar allí. Se presentan como el primer método de *open-world sound morphing* sobre difusión preentrenada.

**Tipo de morphing y uniformidad perceptual.** *Open-world*: timbre de instrumentos, sonidos ambientales y música. Soporta las tres modalidades clásicas (static / dynamic / cyclostationary). Su aporte conceptual es el **morphing perceptualmente uniforme**: en lugar de asumir que la percepción es lineal en el factor α, buscan una **trayectoria** en la que cada paso produzca una diferencia perceptual **constante**. Para ello definen la métrica **SPDP** (*Sound Perceptual Distance Proportion*) sobre el log-mel-espectrograma y, mediante **búsqueda binaria**, hallan la secuencia **no lineal** de factores {αᵢ} que iguala esas diferencias.

**Método.** (1) codifica *source* y *target* con el VAE y optimiza sus *embeddings* condicionales por inversión textual; (2) SLERP sobre el ruido `z_T` y LERP sobre los *embeddings*; (3) *denoising* + decoder + vocoder → audio; (4) LoRA para suprimir variación espuria; (5) búsqueda binaria sobre SPDP para lograr el Δp constante. Evalúan con FAD/FID, CDPAM y un criterio de correspondencia (MFCC), comparando contra la Sound Morphing Toolbox (modelo sinusoidal) y MorphFader.

**Ventajas.** Primer método *open-world* sobre difusión preentrenada; **uniformidad perceptual explícita**; versátil (static/dynamic/cyclostationary); guiado por el audio *target* real; marco de evaluación objetivo comprehensivo.

**Desventajas.** Calidad limitada a 16 kHz; depende de un modelo grande multi-componente; falla con sonidos cercanos a ruido o con estructuras temporales muy distintas; **alto costo y latencia** (inversión textual + LoRA + búsqueda binaria que repite el *denoising* por cada candidato α) → **no es tiempo real**.

**Diferencias con nuestra propuesta.**
- Interpola en el **ruido/latente** y en los *embeddings* de texto de **un** modelo; nosotros en los **pesos** de modelos independientes.
- Modelo grande preentrenado con costo por par vs. VAE chico entrenado una vez por instrumento e interpolación de pesos directa y barata.
- Persigue **uniformidad perceptual** (α no lineal); nosotros no la buscamos explícitamente.
- No tiempo real / 16 kHz vs. nuestro objetivo de tiempo real / 22050 Hz.
- **Coincidencia**: ambos usamos FAD (aunque ellos con *embeddings* propios y nosotros con MERT) y SLERP (ellos sobre ruido, nosotros sobre pesos).

**Relevancia.** Estado del arte reciente en *sound morphing* y ejemplo por excelencia del paradigma "modelo generativo grande + interpolación en espacio latente/perceptual con métricas perceptuales formales", contra el cual contrastamos nuestro enfoque liviano.

### 3.3 Chen, Chen, Yu & Ding (2025) — *Guitar Tone Morphing by Diffusion-Based Model* (APSIPA ASC 2025)

**Enfoque.** Compara **cuatro métodos**: tres basados en difusión latente (AudioLDM2 / MusicLDM) con inversión textual y **fine-tuning con LoRA** (incluida una variante *dual-sided* que afina una U-Net por tono y luego **interpola sus pesos con LERP**), y un cuarto **sin difusión** basado en el autoencoder de consistencia **Music2Latent** + SLERP en el latente. **El método ganador es el más simple** (Music2Latent + SLERP): supera en calidad perceptual (MOS 4.3) a los *pipelines* de difusión con *fine-tuning*.

**Tipo de morphing.** Morphing de **"tono"/timbre de guitarra eléctrica** — transiciones entre configuraciones de efectos/amplificación de una misma guitarra (limpio ↔ *high gain*, ↔ modulación, etc.), **no entre instrumentos distintos**. Se formula como interpolación de mel-espectrogramas preservando el contenido musical.

**Método.** SLERP/LERP en el latente + AdaIN para alinear estilo; la variante *dual-LoRA* fusiona pesos de dos U-Nets. Dataset propio de grabaciones reales de guitarra con efectos (5 tareas × 20 pares, clips ~5 s), mayormente a 16 kHz (Music2Latent soporta 44.1 kHz nativo).

**Ventajas.** El método ganador es simple, sin *fine-tuning* ni texto, más robusto y de mejor calidad perceptual; soporta 44.1 kHz.

**Desventajas.** Dominio muy acotado (solo guitarra); los *pipelines* de difusión son complejos y frágiles y el *fine-tuning* con LoRA da resultados erráticos; clips cortos y dataset chico; **el tiempo real se menciona como aspiración pero no se mide**.

**Diferencias con nuestra propuesta.**
- Dominio: tonos/efectos de **una misma guitarra** vs. **instrumentos distintos**.
- Mecanismo intermedio: SLERP en el **latente** (la interpolación de pesos, en la variante *dual-LoRA*, es un refinamiento secundario dentro de un *pipeline* de difusión); en nosotros la interpolación de **pesos** es el mecanismo **central**.
- **Coincidencias**: usan SLERP con caída a LERP cuando los vectores son casi paralelos (idéntico a nuestro criterio), y su variante *dual-LoRA* es el punto de contacto más cercano con la idea de interpolar pesos de la red.
- **Punto de apoyo argumental**: su hallazgo de que un modelo simple y liviano (Music2Latent) supera a la difusión pesada respalda directamente nuestra apuesta por VAEs chicos frente a arquitecturas basadas en difusión.

**Relevancia.** Aplicación muy reciente y específica de morphing con difusión; pertinente por compartir la formulación con SLERP/LERP, por tocar la idea de interpolar pesos y por su conclusión pro-modelos-livianos.

### 3.4 Chu et al. (2026) — *Mix2Morph: Learning Sound Morphing from Noisy Mixes* (Adobe Research / Northwestern)

**Enfoque.** Un **transformer de difusión latente texto-a-audio** grande, preentrenado para generar un solo sonido, **fine-tuneado** para producir morphs. La idea central es aprender morphing a partir de **mezclas ruidosas** como datos sustitutos (*surrogate morphs*), evitando la falta de un dataset dedicado. El truco (estilo *Ambient Diffusion*): asignar las mezclas ruidosas **solo a los timesteps altos** de la difusión, de modo que el modelo aprenda el *concepto* de morph (fusionar dos sonidos) sin sobreajustar a los artefactos de la simple superposición, y aporte el detalle fino desde el conocimiento del preentrenamiento.

**Tipo de morphing.** **SFX / sonidos generales** (diseño sonoro), no timbres de instrumentos afinados — argumentan que para esos el DSP clásico ya funciona. Se enfocan en **"sound infusion"**: un morph estático **asimétrico** donde un sonido primario dominante aporta la estructura temporal y un secundario se "infunde" en su timbre. Atacan el **midpoint collapse** (los extremos son estables pero los morphs intermedios se degradan a una mezcla incoherente).

**Método.** Generan datos aumentando mezclas aditivas con alineación temporal (RMS anchoring) y espectral, con *captions* que codifican la asimetría; *fine-tunean* el modelo con esas mezclas solo en *timesteps* altos. Operan a 48 kHz estéreo. Evalúan con métricas nuevas (LCS, *directionality*), correspondencia (FLAM), FAD y estudio subjetivo con 25 oyentes; superan a *baselines* como MorphFader y SoundMorpher.

**Ventajas.** No requiere dataset de morphs; reutiliza un modelo grande preentrenado; ataca el *midpoint collapse*; alta fidelidad (48 kHz estéreo); control semántico por texto; modela y mide la asimetría.

**Desventajas.** Depende de un modelo TTA grande y de **datos propietarios/licenciados** (no reproducible fácilmente); costoso, **no es tiempo real** ni liviano; dominio SFX, no instrumentos; morph **estático y asimétrico** sin un parámetro α de trayectoria continua; salidas cortas (3 s).

**Diferencias con nuestra propuesta.**
- Aprende el morph **dentro de un único modelo grande** guiado por texto; nosotros interpolamos los **pesos** de VAEs pequeños independientes.
- Fuente de los datos: mezclas ruidosas sintéticas como supervisión vs. cada VAE entrenado solo con su instrumento (el híbrido surge de la interpolación de parámetros, sin datos de morph).
- Dominio SFX vs. timbres instrumentales; control por prompt de texto vs. escalar α continuo; offline y pesado vs. liviano y orientado a tiempo real.
- **Coincidencia**: ambos evitamos el "morph a mezcla aditiva incoherente" y ambos usamos FAD.

**Relevancia.** El antecedente más reciente y de mayor escala; contrapunto ideal por representar el extremo opuesto (gran modelo de difusión texto-a-audio, SFX, control por texto) frente a nuestro morphing tímbrico liviano por interpolación de pesos.

---

## 4. Síntesis: dónde se ubica nuestra propuesta

Ordenando los antecedentes por **dónde ocurre la interpolación**, emergen tres paradigmas y el lugar que ocupamos:

1. **En la señal** (Slaney 1996): warping + cross-fade de representaciones espectrales descompuestas. Requiere *matching* explícito, no aprende, offline.
2. **En el espacio latente / de ruido** de un modelo generativo (NSynth 2017; Latent Diffusion Bridges 2024; SoundMorpher 2024; Guitar Tone Morphing 2025; Mix2Morph 2026): es el paradigma dominante. Un único modelo (o modelos por instrumento) fijo, y la interpolación se hace sobre los códigos/ruido que ese modelo consume. En la variante moderna implica modelos grandes preentrenados (difusión), costosos y lejos del tiempo real.
3. **En el espacio de parámetros (pesos)** — **nuestra propuesta**: modelos VAE chicos, especializados e independientes por instrumento, cuyos *pesos* se interpolan (LERP/SLERP) para fabricar *decoders* intermedios que sintetizan timbres híbridos.

Observaciones transversales que justifican el enfoque:

- **Tendencia hacia modelos cada vez más grandes** (WaveNet → difusión latente preentrenada), a costa de la ejecución en tiempo real y en hardware modesto. Nuestra apuesta por modelos livianos es deliberadamente contracorriente, y el propio hallazgo de Chen et al. (2025) — un autoencoder simple supera a la difusión pesada — la respalda.
- **Casi toda la literatura interpola *contenido* (latente/señal) dentro de un modelo fijo.** Interpolar directamente la *parametrización* de modelos distintos es lo que aporta esta tesis; sólo la variante *dual-LoRA* de Chen et al. lo roza, y de forma secundaria y acotada a la guitarra.
- **Convergencia en FAD** como métrica de similitud tímbrica (NSynth usa clasificadores; los trabajos de difusión y esta tesis usan FAD, aunque con distintos *embeddings*), lo que valida nuestra elección de FAD sobre MERT.
- **Distinción morphing vs. timbre transfer**: conviene explicitarla en el texto. Mancusi et al. hacen transferencia total preservando contenido; nosotros (como NSynth, SoundMorpher y, parcialmente, Chen et al.) generamos estados intermedios híbridos controlables por un parámetro.
