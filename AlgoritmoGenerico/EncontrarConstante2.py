import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import tkinter as tk
from tkinter import messagebox, scrolledtext
import tabula

# Cargar el dataset desde el PDF
rutaPdf = 'dataSet.pdf'
dataframes = tabula.read_pdf(rutaPdf, pages='all')
datos = pd.concat(dataframes, ignore_index=True)  # Concatenar todas las páginas en un solo DataFrame

X = datos[['x1', 'x2', 'x3', 'x4']]
y = datos['y']

# Definir la función objetivo
def funcionObjetivo(x1, x2, x3, x4, a, b, c, d, e):
    return a + b * x1 + c * x2 + d * x3 + e * x4

# Evaluar la función de aptitud
def evaluarAptitud(individuo, X, y, maximizar):
    a, b, c, d, e = individuo
    predicciones = funcionObjetivo(X['x1'], X['x2'], X['x3'], X['x4'], a, b, c, d, e)
    error = np.mean(np.abs(y - predicciones))  # Norma del error absoluto |error|
    return -error if maximizar else error  # Maximizar la aptitud minimizando el error

# Crear la población inicial
def crearPoblacionInicial(cantidad):
    poblacion = []
    for _ in range(cantidad):
        individuo = [random.uniform(-10, 10) for _ in range(5)]  # Inicializar a, b, c, d, e
        poblacion.append(individuo)                                          # c, d, e  a, b,
    return poblacion

# Selección de pares
def seleccionarPares(poblacion):
    pares = []
    n = len(poblacion)
    for i in range(n):
        for j in range(i + 1, n):
            pares.append((poblacion[i], poblacion[j]))
    return pares

# Cruza
def cruzar(par):
    punto = random.randint(1, 4)
    hijo1 = par[0][:punto] + par[1][punto:]
    hijo2 = par[1][:punto] + par[0][punto:]
    return hijo1, hijo2

# Mutación
def mutar(individuo, probMutacionGen, probMutacionIndividuo):
    if random.random() < probMutacionIndividuo:
        for i in range(len(individuo)):
            if random.random() < probMutacionGen:
                individuo[i] += random.uniform(-1, 1)  # Pequeño cambio aleatorio
    return individuo

# Poda
def podar(poblacion, maxPoblacion, X, y, maximizar):
    poblacion.sort(key=lambda ind: evaluarAptitud(ind, X, y, maximizar), reverse=maximizar)
    if len(poblacion) > maxPoblacion:
        poblacion = poblacion[:maxPoblacion]
    return poblacion

# Graficar los valores reales y las predicciones del mejor individuo por generación
def graficarGeneracionIndividual(y, predicciones, generacion, carpeta):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y)), y, 'ro-', label='Valores Reales')
    plt.plot(range(len(predicciones)), predicciones, 'bx-', label=f'Predicción Generación {generacion}')
    plt.xlabel('Identificador')
    plt.ylabel('Valor')
    plt.title(f'Valores Reales y Predichos - Generación {generacion}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, f'Generacion_{generacion}.png'))
    plt.close()

# Graficar la evolución del error
def graficarEvolucionError(errores, carpeta):
    plt.figure(figsize=(10, 5))
    plt.plot(errores, label='Norma de Error Absoluto', color='purple')
    plt.xlabel('Generación')
    plt.ylabel('Error Absoluto')
    plt.title('Evolución del Error Absoluto')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta, 'Evolucion_Error.png'))
    plt.close()

# Validar entradas del usuario
def validarEntradas():
    try:
        cantidadGeneraciones = int(entradaCantidadGeneraciones.get())
        probMutacionGen = float(entradaProbMutacionGen.get())
        probMutacionIndividuo = float(entradaProbMutacionIndividuo.get())
        cantidadIndividuos = int(entradaCantidadIndividuos.get())
        maxPoblacion = int(entradaMaxPoblacion.get())

        if not (0 <= probMutacionGen <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del gen debe estar entre 0 y 1.")
            return False
        if not (0 <= probMutacionIndividuo <= 1):
            messagebox.showerror("Error de Validación", "La probabilidad de mutación del individuo debe estar entre 0 y 1.")
            return False
        if cantidadGeneraciones <= 0 or cantidadIndividuos <= 0 or maxPoblacion <= 0:
            messagebox.showerror("Error de Validación", "El número de generaciones, individuos y población máxima deben ser números enteros positivos.")
            return False

        return True
    except ValueError:
        messagebox.showerror("Error de Validación", "Por favor, ingrese valores válidos en todos los campos.")
        return False

# Ejecutar el algoritmo genético
def ejecutarAlgoritmoGenetico():
    if not validarEntradas():
        return

    # Obtener los valores ingresados por el usuario
    cantidadGeneraciones = int(entradaCantidadGeneraciones.get())
    probMutacionGen = float(entradaProbMutacionGen.get())
    probMutacionIndividuo = float(entradaProbMutacionIndividuo.get())
    cantidadIndividuos = int(entradaCantidadIndividuos.get())
    maxPoblacion = int(entradaMaxPoblacion.get())
    maximizar = varMaximizar.get() == 1

    # Crear la carpeta para las gráficas si no existe
    carpetaGraficas = "graficas"
    if not os.path.exists(carpetaGraficas):
        os.makedirs(carpetaGraficas)

    # Crear la población inicial
    poblacion = crearPoblacionInicial(cantidadIndividuos)
    errores = []

    textoResultados.delete('1.0', tk.END)

    # Iterar a través de las generaciones
    for generacion in range(cantidadGeneraciones + 1):
        aptitudes = [evaluarAptitud(ind, X, y, maximizar) for ind in poblacion]
        mejorAptitud = max(aptitudes) if maximizar else min(aptitudes)
        mejorIndividuo = poblacion[aptitudes.index(mejorAptitud)]

        # Calcular el error absoluto del mejor individuo
        a, b, c, d, e = mejorIndividuo
        predicciones = funcionObjetivo(X['x1'], X['x2'], X['x3'], X['x4'], a, b, c, d, e)
        error = np.mean(np.abs(y - predicciones))

        errores.append(error)

        # Graficar los valores reales y las predicciones del mejor individuo de esta generación
        graficarGeneracionIndividual(y, predicciones, generacion, carpetaGraficas)

        # Crear una tabla con los resultados de la generación actual
        tabla = PrettyTable()
        tabla.field_names = ["Generación", "Individuo", "Aptitud", "Error"]
        tabla.add_row([generacion, mejorIndividuo, round(mejorAptitud, 3), round(error, 3)])
        textoResultados.insert(tk.END, tabla.get_string() + "\n")

        # Si no es la última generación, continuar con la cruza y mutación
        if generacion < cantidadGeneraciones:
            pares = seleccionarPares(poblacion)
            nuevaPoblacion = []

            for par in pares:
                if random.random() < 0.7:  # Tasa de cruza
                    descendencia = cruzar(par)
                    nuevaPoblacion.extend(descendencia)
                else:
                    nuevaPoblacion.extend(par)

            nuevaPoblacion = [mutar(ind, probMutacionGen, probMutacionIndividuo) for ind in nuevaPoblacion]
            poblacion = podar(nuevaPoblacion, maxPoblacion, X, y, maximizar)
            poblacion.append(mejorIndividuo)

    poblacion = podar(poblacion, maxPoblacion, X, y, maximizar)

    # Graficar la evolución del error
    graficarEvolucionError(errores, carpetaGraficas)

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Algoritmo Genético")

tk.Label(root, text="Número de Generaciones:").grid(row=0, column=0, sticky=tk.W)
entradaCantidadGeneraciones = tk.Entry(root)
entradaCantidadGeneraciones.grid(row=0, column=1)

tk.Label(root, text="Probabilidad de Mutación del Gen:").grid(row=1, column=0, sticky=tk.W)
entradaProbMutacionGen = tk.Entry(root)
entradaProbMutacionGen.grid(row=1, column=1)

tk.Label(root, text="Probabilidad de Mutación del Individuo:").grid(row=2, column=0, sticky=tk.W)
entradaProbMutacionIndividuo = tk.Entry(root)
entradaProbMutacionIndividuo.grid(row=2, column=1)

tk.Label(root, text="Número de Individuos:").grid(row=3, column=0, sticky=tk.W)
entradaCantidadIndividuos = tk.Entry(root)
entradaCantidadIndividuos.grid(row=3, column=1)

tk.Label(root, text="Población Máxima:").grid(row=4, column=0, sticky=tk.W)
entradaMaxPoblacion = tk.Entry(root)
entradaMaxPoblacion.grid(row=4, column=1)

tk.Label(root, text="Maximizar Función:").grid(row=5, column=0, sticky=tk.W)
varMaximizar = tk.IntVar()
tk.Checkbutton(root, variable=varMaximizar).grid(row=5, column=1, sticky=tk.W)

tk.Button(root, text="Ejecutar", command=ejecutarAlgoritmoGenetico).grid(row=6, column=0, columnspan=2)

textoResultados = scrolledtext.ScrolledText(root, width=160, height=45)
textoResultados.grid(row=7, column=0, columnspan=2)

root.mainloop()