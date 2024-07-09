import os
import random
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from tkinter import Tk, Label, Entry, Button, ttk, StringVar

def poda(poblacion, max_poblacion):
    clase1=[]
    clase2=[]
    clase3=[]
    nueva_poblacion = []
    size1 = 0
    size2 = 0
    size3= 0
    paso1 = False
    paso2 = False
    clase1.append(poblacion[0])
    for i in range(max_poblacion):
        if not paso1:
            size1+=1
            paso1=True
            paso2=False
        elif not paso2:
            size2+=1
            paso2=True
        else:
            size3+=1
            paso1=False
    while size1>len(clase1):
        posicion = random.randint(1, len(poblacion)-1)
        esta= poblacion[posicion]in clase1
        if not esta:
            clase1.append(poblacion[posicion])
    while size2>len(clase2):
        posicion = random.randint(1, len(poblacion)-1)
        esta = poblacion[posicion] in clase1 and poblacion[posicion]in clase2
        if not esta:
                clase2.append(poblacion[posicion])
    while size3>len(clase3):
        posicion = random.randint(1, len(poblacion)-1)
        esta = poblacion[posicion] in clase1 and poblacion[posicion]in clase2 and poblacion[posicion]in clase3
        if not esta:
                clase3.append(poblacion[posicion])
    for individuo in clase1:
        nueva_poblacion.append(individuo)
    for individuo in clase2:
        nueva_poblacion.append(individuo)
    for individuo in clase3:
        nueva_poblacion.append(individuo)
    return nueva_poblacion

def definir_valores():
    generaciones = []
    mejores_de_cada_generacion = []
    promedios_de_cada_generacion = []
    peores_de_cada_generacion = []
    try:
        a = float(entry_a.get())
        b = float(entry_b.get())
        delta = float(entry_delta.get())
        pmutacion = float(p_mutacion.get())
        tgeneracion = int(n_generaciones.get())
        max_poblacion = int(poblacion_maxima.get())
        modo = combo_operacion.get()
        ventana.destroy()

        k=calcular_decimales(str(delta))
        p = ((b - a) / delta + 1)
        bits = round(np.log2(p))
        deltax = round(((b - a) / (2 ** bits)), k+1)
        
        cni = random.randint(2, max_poblacion)
        
        inicio = generar_binario(bits, cni)
        indexs = convertir_decimal(inicio)
        x = definir_x(a, deltax, indexs)
        fx = definir_fx(x)
        crear_graficas(a, b, fx['Fx'], x, 0.1, modo)
        poblacion = (crear_poblaciones(inicio, indexs, x, fx['Fx'], modo))
        reverse = True if modo == "maximizar" else False
        poblacion = sorted(poblacion, key=lambda Fx: Fx['Fx'], reverse=reverse)
        generaciones.append(poblacion)

        fitness = evaluar_fitness(generaciones[0])
        mejores_de_cada_generacion.append(fitness[0])
        promedios_de_cada_generacion.append(fitness[1])
        peores_de_cada_generacion.append(fitness[2])
        print(f"Valores ingresados: a={a}, b={b}, delta={deltax}, bits={bits} puntos={p}")

        j=0.1
        for i in range(1, tgeneracion):
            j+=0.1
            j = round(j, 1)
            poblacion=[]
            poblacion = generar_generaciones(pmutacion, a, b, inicio, deltax,j, max_poblacion, modo)
            generaciones.append(poblacion['poblacion'])
            inicio = poblacion['binario']
            mejores_de_cada_generacion.append(poblacion['fitness'][0])
            promedios_de_cada_generacion.append(poblacion['fitness'][1])
            peores_de_cada_generacion.append(poblacion['fitness'][2])
        
        crear_video()
        crear_tabla(mejores_de_cada_generacion, deltax)
        crear_grafica(mejores_de_cada_generacion, promedios_de_cada_generacion, peores_de_cada_generacion)
    except ValueError:
        print("Error: Por favor ingrese números válidos.")

def calcular_decimales(numero_str):
    if '.' in numero_str:
        return len(numero_str.split('.')[1])
    else:
        return 0
    
def eliminar_duplicados_mantener_orden(arreglo):
    visto = set()
    resultado = []
    for item in arreglo:
        if item not in visto:
            resultado.append(item)
            visto.add(item)
    return resultado

def crear_tabla(datos, delta):
    ventana = Tk()
    ventana.title("Mejor Individuo")
    texto = 'Delta x: '+str(delta)
    Label(ventana, text=texto).grid(row=0, column=0)

    # Crear un Treeview para la tabla
    columnas = ("Generacion", "cadena_bits", "indice", "valor_x", "aptitud")
    tabla = ttk.Treeview(ventana, columns=columnas, show="headings")
    
    # Definir encabezados
    tabla.heading("Generacion", text="Generacion")
    tabla.heading("cadena_bits", text="Cadena de Bits")
    tabla.heading("indice", text="Índice")
    tabla.heading("valor_x", text="Valor de X")
    tabla.heading("aptitud", text="Aptitud")
    
    # Ajustar el ancho de las columnas
    tabla.column("Generacion", width=50)
    tabla.column("cadena_bits", width=150)
    tabla.column("indice", width=50)
    tabla.column("valor_x", width=100)
    tabla.column("aptitud", width=100)
    
    # Insertar datos en la tabla
    for i, dato in enumerate(datos):
        tabla.insert("", "end", values=(i+1, dato["Binario"], dato["Identificador"], dato["X"], dato["Fx"]))
    
    # Colocar la tabla usando grid
    tabla.grid(row=1, column=1, padx=20, pady=20)
    
    # Ejecutar el bucle principal de la ventana
    ventana.mainloop()

def mostrar_ventana():
    global ventana, combo_operacion
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Valor de a:").grid(row=0, column=0)
    Label(ventana, text="Valor de b:").grid(row=1, column=0)
    Label(ventana, text="Valor de delta:").grid(row=2, column=0)
    Label(ventana, text="Valor de probabilidad de mutación del gen:").grid(row=3, column=0)
    Label(ventana, text="Generaciones:").grid(row=4, column=0)
    Label(ventana, text="Población máxima:").grid(row=5, column=0)
    Label(ventana, text="Operación (maximizar/minimizar):").grid(row=6, column=0)

    global entry_a, entry_b, entry_delta, p_mutacion, n_generaciones, poblacion_maxima
    entry_a = Entry(ventana)
    entry_b = Entry(ventana)
    entry_delta = Entry(ventana)
    p_mutacion = Entry(ventana)
    n_generaciones = Entry(ventana)
    poblacion_maxima = Entry(ventana)
    combo_operacion = ttk.Combobox(ventana, values=["maximizar", "minimizar"])
    combo_operacion.current(0)

    entry_a.grid(row=0, column=1)
    entry_b.grid(row=1, column=1)
    entry_delta.grid(row=2, column=1)
    p_mutacion.grid(row=3, column=1)
    n_generaciones.grid(row=4, column=1)
    poblacion_maxima.grid(row=5, column=1)
    combo_operacion.grid(row=6, column=1)

    Button(ventana, text="Aceptar", command=definir_valores).grid(row=7, column=0, columnspan=3)

    ventana.mainloop()

def operacion(x):
    return np.log(abs(0.1 + x**2)) * np.cos(x)**2

def crear_graficas(a, b, fx, x, i, modo):
    img_dir = "gen_images"
    os.makedirs(img_dir, exist_ok=True)

    def save_plots(a, b, generacion_x, generacion_fx, gen_index, modo):
        x = np.linspace(a, b, 1000)
        y = operacion(x)
        
        if modo == 'maximizar':
            max_index = np.argmax(generacion_fx)
            min_index = np.argmin(generacion_fx)
        else:
            min_index = np.argmax(generacion_fx)
            max_index = np.argmin(generacion_fx)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(generacion_x, generacion_fx, color='blue')
        plt.scatter(generacion_x[max_index], generacion_fx[max_index], color='green', s=100, label='Mejor')
        plt.scatter(generacion_x[min_index], generacion_fx[min_index], color='red', s=100, label='Peor')
        plt.plot(x, y, color='blue')
        plt.title(f'Generación {gen_index}')
        plt.xlabel('GeneracionX')
        plt.ylabel('GeneracionFx')
        plt.grid(True)
        plt.legend()
        filename = f"{img_dir}/generation_{gen_index}.png"
        plt.savefig(filename)
        plt.close()
    
    save_plots(a, b, x, fx, i, modo)

def crear_video():
    clip = ImageSequenceClip('gen_images', fps=1)
    video_filename = 'generations_video.mp4'
    clip.write_videofile(video_filename, codec='libx264')

def crear_grafica(mejores_de_cada_generacion, promedios_de_cada_generacion, peores_de_cada_generacion):
    x_mejores = range(len(mejores_de_cada_generacion))
    x_promedio = range(len(promedios_de_cada_generacion))
    x_peores = range(len(peores_de_cada_generacion))
    fx_mejores = [individuo['Fx'] for individuo in mejores_de_cada_generacion]
    fx_peores = [individuo['Fx'] for individuo in peores_de_cada_generacion]
    
    plt.plot(x_mejores, fx_mejores, label='Mejores de cada generación')
    plt.plot(x_promedio, promedios_de_cada_generacion, label='Promedios de cada generación')
    plt.plot(x_peores, fx_peores, label='Peores de cada generación')
    plt.title('Mejores de cada generación')
    plt.xlabel('Generación')
    plt.ylabel('Fx')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
    plt.show()

def evaluar_fitness(arreglo):
    mejor=[]
    promedio=0
    peor=[]
    fx = []
    for individuo in arreglo:
        fx.append(individuo['Fx'])
    mejor = arreglo[0]
    promedio = round(sum(fx) / len(fx), 2)
    peor = arreglo[len(arreglo)-1]
    return [mejor, promedio, peor]

def generar_generaciones(pmutacion, a, b, inicio, delta, j, max_poblacion, modo):
    nuevos = []
    parejas_cruce = []
    poblacion=[]
    parejas_cruce = generar_parejas(inicio)
    nuevos.append(inicio[0])
    for pareja1, parejas in parejas_cruce:
        for pareja2 in parejas:
            hijo1, hijo2 = cruzar_individuos(pmutacion, pareja1, pareja2)
            nuevos.append(hijo1)
            nuevos.append(hijo2)
    nuevos = eliminar_duplicados_mantener_orden(nuevos)
    indexs = convertir_decimal(nuevos)
    xn = definir_x(a, delta, indexs)
    fxn = definir_fx(xn)
    poblacion = crear_poblaciones(nuevos, indexs, xn, fxn['Fx'], modo)
    fitness = evaluar_fitness(poblacion)
    if len(poblacion)>max_poblacion:
        poblacion = poda(poblacion, max_poblacion)
    fx = []
    x = []
    nuevos=[]
    for individuo in poblacion:
        nuevos.append(individuo['Binario'])
        fx.append(individuo['Fx'])
        x.append(individuo['X'])
    crear_graficas(a, b, fx, x, j, modo)
    return {'poblacion': poblacion, 'binario': nuevos, 'fitness':fitness}

def generar_binario(bits, cni):
    inicio = set()
    while len(inicio) < cni:
        nbin = ''.join(random.choice('01') for _ in range(bits))
        inicio.add(nbin)
    return list(inicio)

def intercambiar_bits(individuo, pos1):
    individuo_list=[]
    pos2 = random.randint(0, len(individuo) - 1)
    individuo_list = list(individuo)
    individuo_list[pos1], individuo_list[pos2] = individuo_list[pos2], individuo_list[pos1]
    return ''.join(individuo_list)

def mutacion(individuo, pmutacion):
    nuevo = individuo
    for i, bit in enumerate(nuevo):
        if random.randint(1, 99)/100 < pmutacion:
            nuevo = intercambiar_bits(nuevo, i)
    return nuevo

def crear_poblaciones(inicio, indexs, x, fx, modo):
    poblacion = [{'Binario': binario, 'Identificador': idx, 'X': x_val, 'Fx': fx_val}
                 for binario, idx, x_val, fx_val in zip(inicio, indexs, x, fx)]
    reverse = True if modo == "maximizar" else False
    poblacion = sorted(poblacion, key=lambda Fx: Fx['Fx'], reverse=reverse)
    return poblacion

def generar_parejas(poblacion):
    parejas_cruce = []
    cantidad_parejas = 0
    poblacion_indices = list(range(len(poblacion)))
    
    while(cantidad_parejas<=0):
        for i in range(len(poblacion)):
            cantidad_parejas = random.randint(0, len(poblacion)-1)
            parejas = random.sample(poblacion_indices[:i] + poblacion_indices[i+1:], cantidad_parejas)
            parejas_cruce.append((poblacion[i], [poblacion[j] for j in parejas]))
    
    return parejas_cruce

def cruzar_individuos(pmutacion, individuo1, individuo2):
    punto_de_cruce = random.randint(1, len(individuo1) - 2)
    hijo1 = individuo1[:punto_de_cruce] + individuo2[punto_de_cruce:]
    hijo2 = individuo2[:punto_de_cruce] + individuo1[punto_de_cruce:]
    return mutacion(hijo1, pmutacion), mutacion(hijo2, pmutacion)

def convertir_decimal(inicio):
    indexs = []
    for ninicio in inicio:
        numero = int(ninicio, 2)
        indexs.append(numero)
    return indexs

def definir_x(a, deltax, indexs):
    x_values = [float(a+(idx*deltax)) for idx in indexs]
    return x_values

def definir_fx(x):
    fx = [operacion(xi) for xi in x]
    return {'datos': {'X': x, 'Fx': fx}, 'Fx': fx}

mostrar_ventana()
