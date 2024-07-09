import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, ttk

def main():
    xs, resultados = obtener_variables()
    try:
        pmutacion = float(p_mutacion.get())
        pmutaciong = float(p_mutaciong.get())
        tgeneraciones = int(n_generaciones.get())
        max_poblacion = int(poblacion_maxima.get())
        min_poblacion = int(poblacion_minima.get())
        mejores =[]
        j=0
        while j < len(resultados)-1:
            a=[]
            b=[]
            c=[]
            d=[]
            e=[]
            individuos = []
            individuos_inicial = random.randint(min_poblacion, max_poblacion)
            generaciones = []
            poblacion = []
            fx = []
            yd = []
            i = 0
            usar_xs = [xs[0][j], xs[1][j], xs[2][j], xs[3][j]]
            while i < individuos_inicial:
                constantes = crear_individuos()
                y = definir_y(constantes, usar_xs)
                fitnes=fitness(y, resultados[j])
                poblacion.append({'fitness': fitnes, 'Yd': resultados[j], 'Fx': y, 'constantes': constantes})
                i += 1
            
            poblacion = sorted(poblacion, key=lambda fitnes: fitnes['fitness'])
            generaciones.append(poblacion)
            i = 1
            while i < tgeneraciones+1:
                poblacion = crear_generaciones(usar_xs, poblacion, max_poblacion, pmutacion, resultados[j], pmutaciong)
                generaciones.append(poblacion)
                i += 1
            mejores.append(generaciones[len(generaciones)-1][0])
            for p in generaciones:
                # print(p[0])
                individuos.append(p[0]['fitness'])
                for individuo in p:
                    a.append(individuo['constantes'][0])
                    b.append(individuo['constantes'][1])
                    c.append(individuo['constantes'][2])
                    d.append(individuo['constantes'][3])
                    e.append(individuo['constantes'][4])
            # print(individuos)
            crear_graficas(individuos, j+1)
            crear_graficas_constante(a, b, c, d, e, j)
            j+=1
        mostrar_tabla(mejores)
        for mejor in mejores:
            fx.append(mejor['Fx'])
            yd.append(mejor['Yd'])
        crear_grafica(yd, fx)
    except ValueError:
        print('datos erroneos')

def crear_graficas(fx, i):
    img_dir = "gen_images_fitness"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(generacion_fx, gen_index):
        plt.figure(figsize=(8, 6))
        plt.plot(generacion_fx, color='blue', label='Generacion norma de error')
        plt.title(f'Generación {gen_index}')
        plt.xlabel('Norma de error')
        plt.ylabel('Generación')
        plt.grid(True)
        plt.legend()
        filename = f"{img_dir}/generation{gen_index}.png"
        plt.savefig(filename)
        plt.close()
    
    save_plots(fx, i)

def crear_graficas_constante(a, b, c, d, e, i):
    img_dir = "gen_images_constantes"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, c, d, e, gen_index):
        x_a = np.linspace(0, len(a)-1, len(a))
        x_b = np.linspace(0, len(b)-1, len(b))
        x_c = np.linspace(0, len(c)-1, len(c))
        x_d = np.linspace(0, len(d)-1, len(d))
        x_e = np.linspace(0, len(e)-1, len(e))
        plt.figure(figsize=(8, 6))
        plt.scatter(x_a, a, color='blue', label='A')
        plt.plot(a, color='blue', label='A')
        plt.scatter(x_b, b, color='red', label='B')
        plt.plot(b, color='red', label='A')
        plt.scatter(x_c, c, color='green', label='C')
        plt.plot(c, color='green', label='A')
        plt.scatter(x_d, d, color='black', label='D')
        plt.plot(d, color='black', label='A')
        plt.scatter(x_e, e, color='skyblue', label='E')
        plt.plot(e, color='skyblue', label='A')
        plt.title(f'Generación {gen_index}')
        plt.xlabel('Valor constante')
        plt.ylabel('Generación')
        plt.grid(True)
        plt.legend()
        filename = f"{img_dir}/generation{gen_index}.png"
        plt.savefig(filename)
        plt.close()
    
    save_plots(a, b, c, d, e, i)

def obtener_variables():
    file_path = '2024.05.22 dataset 8A.csv'
    df = pd.read_csv(file_path)

    # Obtener las columnas específicas
    columnas_interes = ['x1', 'x2', 'x3', 'x4']
    datos = df[columnas_interes]
    resultados = df['y']
    xs = []
    for dato in datos:
        xs.append(datos[dato])
    return xs, resultados

def mostrar_ventana():
    global ventana
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Valor de probabilidad de mutación del individuo:").grid(row=1, column=0)
    Label(ventana, text="Valor de probabilidad de mutación del gen:").grid(row=2, column=0)
    Label(ventana, text="Generaciones:").grid(row=3, column=0)
    Label(ventana, text="Población máxima:").grid(row=4, column=0)
    Label(ventana, text="Población mínima:").grid(row=5, column=0)

    global p_mutacion, n_generaciones, poblacion_maxima, poblacion_minima, p_mutaciong, treeview
    p_mutacion = Entry(ventana)
    p_mutaciong = Entry(ventana)
    n_generaciones = Entry(ventana)
    poblacion_maxima = Entry(ventana)
    poblacion_minima = Entry(ventana)

    p_mutacion.grid(row=1, column=1)
    p_mutaciong.grid(row=2, column=1)
    n_generaciones.grid(row=3, column=1)
    poblacion_maxima.grid(row=4, column=1)
    poblacion_minima.grid(row=5, column=1)

    Button(ventana, text="Aceptar", command=main).grid(row=6, column=0, columnspan=3)

    # Crear un Treeview para la tabla
    treeview = ttk.Treeview(ventana, columns=("fitness", "Yd", "Fx", "constantes"), show="headings")
    treeview.heading("fitness", text="Fitness")
    treeview.heading("Yd", text='Yd')
    treeview.heading("Fx", text="Fx")
    treeview.heading("constantes", text="Constantes")
    treeview.grid(row=7, column=0, columnspan=3)

    ventana.mainloop()

def crear_grafica(yd, fx):
    x_yd = range(len(yd))
    x_fx = range(len(fx))
    
    plt.plot(x_fx, fx, color='red', label='Resuldato obtenido')
    plt.plot(x_yd, yd, label='Resultado esperado')
    plt.scatter(x_fx, fx, color='red', s= 100, label='Resultado obtenidos')
    plt.scatter(x_yd, yd, color='blue', s= 20, label='Resultado deseados')
    plt.title('Mejores de cada generación')
    plt.xlabel('Generación')
    plt.ylabel('Fx')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.show()

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    for mejor in mejores:
        treeview.insert("", "end", values=(mejor['fitness'], mejor['Yd'], mejor['Fx'], 
                                                                    (mejor['constantes'][0],":" , 
                                                                    mejor['constantes'][1],":" , 
                                                                    mejor['constantes'][2],":" , 
                                                                    mejor['constantes'][3],":" , 
                                                                    mejor['constantes'][4])))

def crear_individuos():
    constantes = [round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2),
                  round(float(random.random()), 2)]
    return constantes

def definir_y(constante, xs):
    y = 0
    i = 0
    for a in constante:
        if y == 0:
            y += a
        else:
            y += a * xs[i]
            i += 1
    return round(y, 2)

def cruza(pareja1, pareja2):
    constantes1 = pareja1['constantes']
    constantes2 = pareja2['constantes']
    posicion = random.randint(1, len(constantes1) - 1)
    hijo1 = constantes1[:posicion] + constantes2[posicion:]
    hijo2 = constantes2[:posicion] + constantes1[posicion:]
    return hijo1, hijo2

def fitness(y, resultado):
    return np.linalg.norm(round(abs(resultado - y), 2))

def mutacion(individuo, pmutacion):
    nuevo = individuo
    muto = False
    while not muto:
        for i, constante in enumerate(nuevo):
            if random.randint(1, 99) / 100 < pmutacion:
                nuevo[i] = round(constante * (1 + (np.random.normal(0, 0.4))), 2)
                muto = True
    return nuevo

def definir_mutacion(hijo, prabilidad_mutacioni, prabilidad_mutaciong):
    if random.randint(1, 99) / 100 < prabilidad_mutacioni:
        hijo = mutacion(hijo, prabilidad_mutaciong)
    return hijo

def crear_generaciones(xs, poblacion, max_individuos, prabilidad_mutacioni, resultado, prabilidad_mutaciong):
    nuevos = []
    nueva_poblacion = []
    nueva_poblacion.append(poblacion[0])
    parejas_cruce = generar_parejas(poblacion)
    for pareja1, parejas in parejas_cruce:
        for pareja2 in parejas:
            hijo1, hijo2 = cruza(pareja1, pareja2)
            nuevos.append(definir_mutacion(hijo1, prabilidad_mutacioni, prabilidad_mutaciong))
            nuevos.append(definir_mutacion(hijo2, prabilidad_mutacioni, prabilidad_mutaciong))
    for constantes in nuevos:
        y = definir_y(constantes, xs)
        fitnes = fitness(y, resultado)
        nueva_poblacion.append({'fitness': fitnes, 'Yd': resultado, 'Fx': y, 'constantes': constantes})
    nueva_poblacion = sorted(nueva_poblacion, key=lambda fitnes: fitnes['fitness'])
    nueva_poblacion = podar(nueva_poblacion, max_individuos)
    return nueva_poblacion

def podar(poblacion, max_individuos):
    nueva_poblacion = []
    i = 0
    while len(nueva_poblacion) < max_individuos and i < len(poblacion):
        nueva_poblacion.append(poblacion[i])
        i += 1
    return nueva_poblacion

def generar_parejas(poblacion):
    parejas_cruce = []
    poblacion_indices = list(range(len(poblacion)))
    
    for i in range(len(poblacion)):
        cantidad_parejas = random.randint(0, len(poblacion)-1)
        parejas = random.sample(poblacion_indices[:i] + poblacion_indices[i + 1:], cantidad_parejas)
        parejas_cruce.append((poblacion[i], [poblacion[j] for j in parejas]))
    
    return parejas_cruce

mostrar_ventana()
