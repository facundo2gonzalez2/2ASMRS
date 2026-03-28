Experimentos y gráficos para la investigación de VAE e interpolación:

1. Arquitectura del modelo: objetivo definir el tamaño del espacio latente  
   a. Error de Validación vs Tamaño del espacio latente por instrumento.
   b. Mismo gráfico pero promediando y agregando el error estadístico.


2. Comparación beta-vae:  
   a. Error de reconstrucción (MSE) a lo largo del entrenamiento para el modelo `base_model` con 5 betas diferentes: Observamos que beta=0.01 penaliza demasiado el MSE y elegimos como beta para el resto de la tesis beta=0.001 (balance entre no penalizar MSE e intentar no tener mucha divergencia KL).

   b. Divergencia KL vs espacio latente para beta-vae por instrumento.  
   c. Comparación de error de reconstrucción validación (MSE) vs espacio latente para beta \= 0.001 en cada instrumento: Observamos que para obtener el mismo MSE que sin vae, hay que aumentar el espacio latente (son la misma curva pero desplazada para arriba)


3. Análisis del espacio latente con UMAP  
   a. Usando el `base_model` para reconstruir 1 track por instrumento (¿repetir con varios tracks por instrumento?) visualizar nube de puntos del espacio latente. Buscamos observamos cierta separación/clustering dentro del espacio latente por instrumento, quizás teniendo algunos instrumentos más cerca entre sí que otros. **Para sumar: graficar métrica de distancia entre clusters.** actualmente solo imprime los datos.


4. Interpolación  
   a. Similitud de reconstrucción vs alfa mirando 1 solo instrumento: A medida que nos acercamos al modelo de {instr} va mejorando la similitud de reconstrucción entre varios tracks de ese audio (por ej: interpolar guitarra y piano, reconstrucción sobre varios tracks de piano y a medida que aumenta alfa mejora la reconstrucción). Con similitud coseno y distancia FAD (convertir distancia FAD a similitud FAD).

   b. Similitud de reconstrucción en función del alfa: queremos ver que se acerca de un instrumento a otro y se mantiene neutral frente a otro testigo. Repetir para varios pares de instrumentos. Diferencias entre from\_checkpoint y from\_scratch y beta vs no beta

   c. ***Para sumar: algún gráfico con métricas sobre distancia a centroides haciendo el UMAP de 2 dimensiones*** actualmente se imprimen los datos usando multidimensional_ranksum.