import matplotlib.pyplot as plt
import numpy as np

# Datos para los gráficos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.cos(x)

# Crear una figura y un conjunto de subgráficos
fig, axs = plt.subplots(2, 3)  # 1 fila, 2 columnas

# Primer subgráfico
axs[0, 0].plot(x, y1)
axs[0, 0].set_title('Seno de x')

# Segundo subgráfico
axs[0, 1].plot(x, y2)
axs[0, 1].set_title('Coseno de x')

axs[0, 2].plot(x, y3)
axs[0, 2].set_title('Coseno de x')

# Mostrar la figura
plt.tight_layout()  # Ajusta automáticamente los parámetros de la subtrama
plt.show()