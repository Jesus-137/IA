from tkinter import Tk, Label, Entry, Button, ttk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

def obtener_variables():
    file_path = '2024.05.22 dataset 8A.csv'
    df = pd.read_csv(file_path, delimiter='; ')

    datos = df[['x1', 'x2', 'x3', 'x4', 'x5']]
    resultados = df['y'].values 

    escalar = StandardScaler()
    datos_escalados = escalar.fit_transform(datos)

    return datos_escalados, resultados

def agregar_sesgo(x):
    x0 = np.ones((x.shape[0], 1))
    return np.hstack((x0, x))

def inicializar_pesos(n):
    return np.random.uniform(-0.5, 0.5, n)

def verificar_inf_nan(valores):
    return np.isinf(valores).any() or np.isnan(valores).any()

def U(x, w):
    u = np.dot(x, w)
    if verificar_inf_nan(u):
        print("Overflow o NaN en U")
        return np.full(u.shape, float('inf'))
    return u

def calcular_error(yc, yd):
    return yd - yc

def escalar_1_0(y, yd):
    y_min = np.min(yd)
    y_max = np.max(yd)
    divisor = y_max - y_min
    if divisor == 0:
        return np.zeros_like(y)
    return (y - y_min) / divisor

def delta_w(tasa_aprendizaje, x, error):
    delta_w = tasa_aprendizaje * np.dot(x.T, error)
    if verificar_inf_nan(delta_w):
        print("Overflow o NaN en delta_w")
        return np.full(delta_w.shape, float('inf'))
    return delta_w

def actualizar_pesos(w, delta_w):
    return w + delta_w

def funcion_activacion(u):
    for i, y in enumerate(u):
        if y <=0:
            u[i]=0
        else:
            u[i]=1
    return u

def principal():
    tasa_aprendizaje = float(t_aprendizaje.get())
    iteraciones = int(epocas.get())
    X, Y = obtener_variables()
    X = agregar_sesgo(X)
    Y = escalar_1_0(Y, Y)
    w = np.array(inicializar_pesos(X.shape[1]))
    WS = []
    norma_errores = []

    for vuelta in range(iteraciones):
        WS.append(w.copy())
        u = U(X, w)
        yc = funcion_activacion(u)
        errores = calcular_error(yc, Y)
        norm_error = np.linalg.norm(errores)
        norma_errores.append(norm_error)

        if norm_error <= 0:
            print(vuelta)
            break

        delta_w_val = delta_w(tasa_aprendizaje, X, errores)
        w = actualizar_pesos(w, delta_w_val)

        if verificar_inf_nan(errores):
            print("Overflow o NaN en errores, parando.")
            break

    crear_grafica_y(yc, Y)
    print(w, norm_error)
    crear_grafica(np.array(WS).T)
    crear_grafica_norma(norma_errores)
    mostrar_tabla({"constantes": w, 'error': norm_error})

def crear_grafica_norma(normas):
    img_dir = "Evidencias"
    os.makedirs(img_dir, exist_ok=True)
    
    plt.plot(normas, label='Norma de error')
    plt.title('Norma de error')
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma')
    plt.grid(True)
    plt.legend(loc='uppercenter', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=1)
    filename = f"{img_dir}/norma_error.png"
    plt.savefig(filename)
    plt.show()

def crear_grafica_y(yc, yd):
    img_dir = "Evidencias"
    os.makedirs(img_dir, exist_ok=True)
    
    plt.plot(yc, label='yc')
    plt.plot(yd, label='yd')
    plt.title('yc, yd')
    plt.xlabel('Iteraciones')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=2)
    filename = f"{img_dir}/yc_yd.png"
    plt.savefig(filename)
    plt.show()

def crear_grafica(constantes):
    img_dir = "Evidencias"
    os.makedirs(img_dir, exist_ok=True)

    labels=['Bias', 'x1', 'x2', 'x3', 'x4', 'x5']
    x = range(len(constantes[0]))
    
    for i, constante in enumerate(constantes):
        plt.plot(x, constante, label=labels[i])
    plt.title('Constantes')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valores')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=len(labels))
    filename = f"{img_dir}/constantes.png"
    plt.savefig(filename)
    plt.show()

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    treeview.insert("", "end", values=(mejores['error'], mejores['constantes']))

def mostrar_ventana():
    global ventana, t_aprendizaje, epocas, treeview
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Tasa de aprendizaje").grid(row=1, column=0)
    Label(ventana, text="Épocas").grid(row=2, column=0)

    t_aprendizaje = Entry(ventana)
    epocas = Entry(ventana)

    t_aprendizaje.grid(row=1, column=1)
    epocas.grid(row=2, column=1)

    Button(ventana, text="Aceptar", command=principal).grid(row=3, column=0, columnspan=2)

    treeview = ttk.Treeview(ventana, columns=("Error", "Constantes"), show="headings")
    treeview.heading("Error", text="Error")
    treeview.heading("Constantes", text="Constantes")
    treeview.grid(row=4, column=0, columnspan=2)

    ventana.mainloop()

mostrar_ventana()
