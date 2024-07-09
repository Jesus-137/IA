import os
import random
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from tkinter import Tk, Label, Entry, Button, ttk

def main():
    a = float(entry_a.get())
    b = float(entry_b.get())
    delta = float(entry_delta.get())
    probabilidadMutacion = float(p_mutacion.get())
    probabilidadMutacioni = float(p_mutacioni.get())
    probabilidadDeCruza = float(p_cruza.get())
    totalGeneracion = int(n_generaciones.get())
    maxPoblacion = int(poblacion_maxima.get())
    minPoblacion = int(poblacion_minima.get())
    modo = combo_operacion.get()

    p = ((b - a) / delta + 1)
    bits = round(np.log2(p))
    deltaX = round(((b - a) / (2 ** bits)), 2)

    generaciones=[]
    mejores = []
    promedios = []
    peores = []
    poblacionInicial = random.randint(minPoblacion, maxPoblacion)
    cantidadPuntos = random.randint(1, bits-1)
    puntosCruza = []
    while len(puntosCruza)<cantidadPuntos:
        punto = random.randint(1, bits-1)
        if not punto in puntosCruza:
            puntosCruza.append(punto)
    individuos = crearIndividuo(poblacionInicial, bits)
    while len(generaciones)<totalGeneracion:
        print(len(generaciones)+1)
        decimales=[Decimal(individuo) for individuo in individuos]
        xs = X(a, deltaX, decimales)
        ys = Y(xs)
        poblacion = CrearPoblaciones(individuos, decimales, xs, ys, modo)
        fxs=[]
        xs=[]
        for individou in poblacion:
            xs.append(individou['X'])
            fxs.append(individou['Fx'])
        mejor, promedio, peor = Fitness(fxs)
        mejores.append(mejor)
        promedios.append(promedio)
        peores.append(peor)
        crear_graficas(a, b, fxs, xs, len(generaciones)+1, modo)
        parejas = A2(individuos, probabilidadDeCruza)
        individuos=[]
        individuos.append(poblacion[0]['Binario'])
        for pareja1, pareja2 in parejas:
            hijo1, hijo2 = C3(pareja1, pareja2, puntosCruza)
            individuos.append(M2(hijo1, probabilidadMutacion, probabilidadMutacioni))
            individuos.append(M2(hijo2, probabilidadMutacion, probabilidadMutacioni))
        individuos = P2(individuos, maxPoblacion)
        generaciones.append(poblacion)
    crearVideo()
    crear_grafica(mejores, promedios, peores)
    mostrar_tabla(poblacion[0])

def crear_grafica(mejores, promedios, peores):
    x_mejores = range(len(mejores))
    x_promedio = range(len(promedios))
    x_peores = range(len(peores))
    
    plt.plot(x_mejores, mejores, label='Mejores de cada generación')
    plt.plot(x_promedio, promedios, label='Promedios de cada generación')
    plt.plot(x_peores, peores, label='Peores de cada generación')
    plt.title('Mejores de cada generación')
    plt.xlabel('Generación')
    plt.ylabel('Fx')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
    plt.show()

def crearVideo():
    clip = ImageSequenceClip('generaciones', fps=1)
    video_filename = 'generations_video.mp4'
    clip.write_videofile(video_filename, codec='libx264')

def crear_graficas(a, b, fx, x, i, modo):
    img_dir = "generaciones"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, generacion_x, generacion_fx, gen_index, modo):
        x = np.linspace(a, b, 1000)
        y = Fx(x)
        
        if modo == 'maximizar':
            max_index = np.argmax(generacion_fx)
            min_index = np.argmin(generacion_fx)
        else:
            min_index = np.argmax(generacion_fx)
            max_index = np.argmin(generacion_fx)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(generacion_x, generacion_fx, color='black')
        plt.scatter(generacion_x[max_index], generacion_fx[max_index], color='green', s=100, label='Mejor')
        plt.scatter(generacion_x[min_index], generacion_fx[min_index], color='red', s=100, label='Peor')
        plt.plot(x, y, color='yellow')
        plt.title(f'Generacion{gen_index}')
        plt.xlabel('X')
        plt.ylabel('Fx')
        plt.grid(True)
        plt.legend()
        filename = f"{img_dir}/generation{gen_index}.png"
        plt.savefig(filename)
        plt.close()
    
    save_plots(a, b, x, fx, i, modo)

def CrearPoblaciones(inicio, indexs, x, fx, modo):
    poblacion = [{'Binario': binario, 'Identificador': idx, 'X': x_val, 'Fx': fx_val}
                 for binario, idx, x_val, fx_val in zip(inicio, indexs, x, fx)]
    reverse = True if modo == "maximizar" else False
    poblacion = sorted(poblacion, key=lambda Fx: Fx['Fx'], reverse=reverse)
    poblacion = PodarDublicados(poblacion)
    return poblacion

def mostrarVentana():
    global ventana, combo_operacion
    ventana = Tk()
    ventana.title("Ingrese valores")
    Label(ventana, text="A:").grid(row=0, column=0)
    Label(ventana, text="B:").grid(row=1, column=0)
    Label(ventana, text="Delta:").grid(row=2, column=0)
    Label(ventana, text="Probabilidad de mutación del gen:").grid(row=3, column=0)
    Label(ventana, text="Probabilidad de mutación del individuo:").grid(row=4, column=0)
    Label(ventana, text="Probabilidad de cruza:").grid(row=5, column=0)
    Label(ventana, text="Generaciones:").grid(row=6, column=0)
    Label(ventana, text="Población máxima:").grid(row=7, column=0)
    Label(ventana, text="Población minima:").grid(row=8, column=0)
    Label(ventana, text="Operación (maximizar/minimizar):").grid(row=9, column=0)
    global treeview, entry_a, entry_b, entry_delta, p_mutacion, p_mutacioni, n_generaciones, poblacion_maxima, poblacion_minima, p_cruza
    entry_a = Entry(ventana)
    entry_b = Entry(ventana)
    entry_delta = Entry(ventana)
    p_mutacion = Entry(ventana)
    p_mutacioni = Entry(ventana)
    p_cruza = Entry(ventana)
    n_generaciones = Entry(ventana)
    poblacion_maxima = Entry(ventana)
    poblacion_minima = Entry(ventana)
    combo_operacion = ttk.Combobox(ventana, values=["maximizar", "minimizar"])
    combo_operacion.current(0)
    entry_a.grid(row=0, column=1)
    entry_b.grid(row=1, column=1)
    entry_delta.grid(row=2, column=1)
    p_mutacion.grid(row=3, column=1)
    p_mutacioni.grid(row=4, column=1)
    p_cruza.grid(row=5, column=1)
    n_generaciones.grid(row=6, column=1)
    poblacion_maxima.grid(row=7, column=1)
    poblacion_minima.grid(row=8, column=1)
    combo_operacion.grid(row=9, column=1)
    Button(ventana, text="Aceptar", command=main).grid(row=10, column=0, columnspan=3)
    treeview = ttk.Treeview(ventana, columns=("Binario", "Decimal", "X", "Fx"), show="headings")
    treeview.heading("Binario", text="Binario")
    treeview.heading("Decimal", text="Decimal")
    treeview.heading("X", text="X")
    treeview.heading("Fx", text="Fx")
    treeview.grid(row=11, column=0, columnspan=3)
    ventana.mainloop()

def mostrar_tabla(mejores):
    for item in treeview.get_children():
        treeview.delete(item)
    treeview.insert("", "end", values=(mejores['Binario'], mejores['Identificador'], mejores['X'], mejores['Fx']))

def A2(poblacion, probabilidadPareja):
    parejas =[]
    for posicion1 in range(len(poblacion)):
        for posicion2 in range(len(poblacion)):
            if posicion1 != posicion2:
                if (random.randint(1, 99)/100) < probabilidadPareja:
                    parejas.append([poblacion[posicion1], poblacion[posicion2]])
    return parejas

def C3 (padre1, padre2, points):
    points = sorted(points)
    points = [0] + points + [len(padre1)]
    
    child1 = []
    child2 = []
    
    for i in range(len(points) - 1):
        if i % 2 == 0:
            child1.append(padre2[points[i]:points[i+1]])
            child2.append(padre1[points[i]:points[i+1]])
        else:
            child1.append(padre1[points[i]:points[i+1]])
            child2.append(padre2[points[i]:points[i+1]])
    
    return ''.join(child1), ''.join(child2)

def M2 (individuo, pmutacion, probabilidad_mutacioni):
    nuevo = individuo
    muto = False
    if random.randint(1, 99) / 100 < probabilidad_mutacioni:
        while not muto:
            for i, bit in enumerate(nuevo):
                if (random.randint(1, 99) / 100) < pmutacion:
                    nuevo = intercambiarBits(nuevo, i)
                    muto = True
    return nuevo

def intercambiarBits(individuo, pos1):
    individuoList=[]
    pos2 = random.randint(1, len(individuo) - 1)
    while pos1 == pos2:
        pos2 = random.randint(1, len(individuo)-1)
    individuoList = list(individuo)
    individuoList[pos1], individuoList[pos2] = individuoList[pos2], individuoList[pos1]
    return ''.join(individuoList)

def P2(poblacion, tamalloMaximoPoblacion):
    nuevaPoblacion = []
    if len(poblacion)>tamalloMaximoPoblacion:
        nuevaPoblacion.append(poblacion[0])
        while len(nuevaPoblacion)<tamalloMaximoPoblacion:
            posicion = int(random.randint(1, len(poblacion)-1))
            esta = poblacion[posicion] in nuevaPoblacion
            while esta:
                posicion+=1
                if posicion>=len(poblacion):
                    posicion=1
                esta = poblacion[posicion] in nuevaPoblacion
            nuevaPoblacion.append(poblacion[posicion])
    else:
        nuevaPoblacion = poblacion
    return nuevaPoblacion

def crearIndividuo(individuosIniciales, bits):
    inicio = set()
    while len(inicio) < individuosIniciales:
        individuo = ''.join(random.choice('01') for _ in range(bits))
        inicio.add(individuo)
    return list(inicio)

def Fx(x):
    return np.log(abs(0.1 + x**2)) * np.cos(x)**2

def Y(x):
    return [Fx(xi) for xi in x]

def X(a, deltaX, ids):
    return [float(a+(id*deltaX)) for id in ids]

def Decimal(binario):
    return int(binario, 2)

def PodarDublicados(poblacion):
    if not poblacion:
        return poblacion

    if isinstance(poblacion[0], dict):
        # Lista de diccionarios
        seen = set()
        nueva_lista = []
        for d in poblacion:
            t = tuple(sorted(d.items()))
            if t not in seen:
                seen.add(t)
                nueva_lista.append(d)
        return nueva_lista
    elif isinstance(poblacion[0], str):
        # Lista de cadenas de bits
        return list(dict.fromkeys(poblacion))
    else:
        raise ValueError("La lista debe contener diccionarios o cadenas de bits.")

def Fitness(fx):
    mejor = fx[0]
    promedio = sum(fx)/len(fx)
    peor = fx[len(fx)-1]
    return mejor, promedio, peor

mostrarVentana()