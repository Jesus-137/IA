import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.preprocessing import StandardScaler

# Global variables
normaError = []
ePesos = []
yc_values_epoca = []
yd_values = []

def funcion_activacion(u):
    for i, y in enumerate(u):
        if y <=0:
            u[i]=0
        else:
            u[i]=1
    return u

def entrenamiento(archivo, epocas, eta):
    global ePesos, normaError, yc_values_epoca, yd_values

    normaError = []
    ePesos = []
    yc_values_epoca = []
    yd_values = []

    data = pd.read_csv(archivo, delimiter='; ', header=0).astype(float)
    X = data.iloc[:, :-1].values
    escalar = StandardScaler()
    X = escalar.fit_transform(X)
    yd = np.array(data.iloc[:, -1]).reshape(-1, 1)
    yd_values = yd 
    xs = X.shape[1]

    model = Sequential()
    model.add(Dense(units=1, input_dim=xs, kernel_initializer='random_uniform', use_bias=True))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=eta), loss='mean_squared_error')
    wIniciales = [w.copy() for w in model.get_weights()]

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global normaError, ePesos, yc_values_epoca
            
            u = model.predict(X)
            yc = funcion_activacion(u)
            e = yd - yc
            norma_error = np.linalg.norm(e)
            normaError.append(norma_error)
            ePesos.append([w.flatten() for w in model.get_weights()])
            yc_values_epoca.append(yc.copy())
            
            guardar_grafica_salidas(epoch)

    model.fit(X, yd, epochs=epocas, callbacks=[CustomCallback()], verbose=0)

    w_finales = [w.copy() for w in model.get_weights()]
    
    return wIniciales, w_finales



def graficar_norma_error():
    global normaError
    sns.set(style="whitegrid")

    plt.figure(figsize=(9, 6))
    x_range = range(1, len(normaError) + 1)
    sns.lineplot(x=x_range, y=normaError, marker='o')
    plt.title('Evolución de la norma del error |e| por época')
    plt.xlabel('Época')
    plt.ylabel('Norma del error |e|')

    plt.show()

def graficar_evolucion_pesos(epocas):
    global ePesos
    sns.set(style="whitegrid")

    plt.figure(figsize=(9, 6))
    for i in range(len(ePesos[0][0])):
        etiqueta = 'Sesgo' if i == len(ePesos[0][0]) - 1 else f'Peso {i+1}'
        sns.lineplot(x=range(1, epocas + 1), y=[peso[0][i] for peso in ePesos], label=etiqueta)

    plt.title('Evolución de los pesos por época')
    plt.xlabel('Época')
    plt.ylabel('Valor del peso')
    plt.legend()

    plt.show()

def guardar_grafica_salidas(epoca):
    global yc_values_epoca, yd_values

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(yd_values) + 1), yd_values, 'b-o', label='Salida deseada (y_d)', markersize=5)
    plt.plot(range(1, len(yc_values_epoca[epoca]) + 1), yc_values_epoca[epoca], 'r-s', label='Salida calculada (y_c)', markersize=5)

    plt.title(f'Comparación de salidas deseadas y calculadas - Época {epoca + 1}')
    plt.xlabel('ID de la observación')
    plt.ylabel('Valor de salida')
    plt.legend()

    plt.close()

def graficar_salidas():
    global yc_values_epoca, yd_values
    
    if not yc_values_epoca or len(yc_values_epoca) == 0:
        print("Primero debe ejecutar el entrenamiento para obtener las salidas calculadas.")
        return

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(yd_values) + 1), yd_values, 'b-o', label='Salida deseada (y_d)', markersize=5)
    plt.plot(range(1, len(yc_values_epoca[-1]) + 1), yc_values_epoca[-1], 'r-s', label='Salida calculada (y_c)', markersize=5)

    plt.title('Comparación de salidas deseadas y calculadas - Última Época')
    plt.xlabel('ID de la observación')
    plt.ylabel('Valor de salida')
    plt.legend()

    plt.show()

def procesar_archivo(archivo, eta, epoca):
    resultado = f"eta={eta} \nepoca={epoca}\n"
    df = pd.read_csv(archivo)
    print("Iniciando procesamiento del archivo CSV...")
    print(df)

    return resultado

print(entrenamiento('2024.05.22 dataset 8A.csv', 335, 0.0001))
graficar_evolucion_pesos(335)
graficar_salidas()