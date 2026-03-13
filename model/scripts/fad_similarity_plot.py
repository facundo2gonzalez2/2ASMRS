import matplotlib.pyplot as plt

# Datos proporcionados
"""
 Alpha 0.00: Similitud = 0.8391, FAD = 78.2161
 Alpha 0.25: Similitud = 0.8334, FAD = 72.2942
 Alpha 0.50: Similitud = 0.8436, FAD = 65.0704
 Alpha 0.75: Similitud = 0.9003, FAD = 49.3675
 Alpha 1.00: Similitud = 0.9627, FAD = 31.6396
"""
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
similarities = [0.8391, 0.8334, 0.8436, 0.9003, 0.9627]
fads = [78.2161, 72.2942, 65.0704, 49.3675, 31.6396]

# Crear la figura y el primer eje (para Similitud)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot de Similitud (Eje Y izquierdo)
color1 = "tab:blue"
ax1.set_xlabel("Alfa", fontsize=12)
ax1.set_ylabel("Similitud de Coseno (MERT)", color=color1, fontsize=12)
line1 = ax1.plot(
    alphas, similarities, marker="o", color=color1, linewidth=2, label="Similitud MERT"
)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(alphas)

# Anotar valores de similitud
for x, y in zip(alphas, similarities):
    ax1.annotate(
        f"{y:.4f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color=color1,
        fontsize=9,
        fontweight="bold",
    )

# Crear el segundo eje (para FAD) compartiendo el mismo eje X
ax2 = ax1.twinx()

# Plot de FAD (Eje Y derecho)
color2 = "tab:red"
ax2.set_ylabel("Distancia FAD", color=color2, fontsize=12)
line2 = ax2.plot(
    alphas, fads, marker="s", color=color2, linewidth=2, linestyle="--", label="FAD"
)
ax2.tick_params(axis="y", labelcolor=color2)

# Anotar valores de FAD
for x, y in zip(alphas, fads):
    ax2.annotate(
        f"{y:.2f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, -15),
        ha="center",
        color=color2,
        fontsize=9,
        fontweight="bold",
    )

# Título y grilla
plt.title("Comparación de Similitud MERT y Distancia FAD vs Alfa", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.6)

# Leyenda unificada
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left")  # type: ignore

# Ajustar márgenes para que los textos no se corten y guardar
fig.tight_layout()
filename = "imgs/similarity_vs_fad.png"
plt.savefig(filename)
print(f"Gráfico guardado como {filename}")
