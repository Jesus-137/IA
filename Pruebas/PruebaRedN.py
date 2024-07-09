import tensorflow as tf
import pandas as pd
from tkinter import Tk, Label, Entry, Button, ttk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Clase de Regresión Lineal
class LinearRegression(tf.keras.Model):
    def __init__(self, lr=0.03):
        super().__init__()
        self.lr = lr
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.dense = tf.keras.layers.Dense(1, kernel_initializer=initializer)

    def call(self, inputs):
        return self.dense(inputs)

# Obtener y escalar datos desde un archivo CSV
def obtener_variables():
    file_path = '2024.05.22 dataset 8A.csv'
    df = pd.read_csv(file_path, delimiter=';')
    datos = df[['x1', 'x2', 'x3', 'x4', 'x5']]
    resultados = df['y'].values 
    escalar = StandardScaler()
    datos = escalar.fit_transform(datos)
    return datos, resultados

# Función para entrenar un modelo de regresión lineal
def entrenar_modelo(tasa_aprendizaje, iteraciones, X, y):
    batch_size = 10
    train_iter = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    model = LinearRegression(lr=tasa_aprendizaje)
    optimizer = tf.keras.optimizers.SGD(learning_rate=model.lr)
    loss = tf.keras.losses.MeanSquaredError()
    
    train_losses = []

    for epoch in range(iteraciones):
        epoch_loss = 0
        for X_batch, y_batch in train_iter:
            with tf.GradientTape() as tape:
                y_hat = model(X_batch, training=True)
                l = loss(y_batch, y_hat)
            grads = tape.gradient(l, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += l.numpy()
        train_losses.append(epoch_loss / len(train_iter))
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_iter):f}')
    
    weights = model.get_weights()
    return weights, train_losses

# Función principal para ejecutar el entrenamiento y mostrar resultados
def principal():
    tasa_aprendizaje = float(t_aprendizaje.get())
    iteraciones = int(epocas.get())
    X, y = obtener_variables()
    weights, train_losses = entrenar_modelo(tasa_aprendizaje, iteraciones, X, y)
    mostrar_resultados(weights, train_losses)

# Mostrar resultados en la interfaz gráfica
def mostrar_resultados(weights, train_losses):
    for item in treeview.get_children():
        treeview.delete(item)
    treeview.insert("", "end", values=(train_losses[-1], weights[0].flatten().tolist(), weights[1].tolist()))

    crear_grafica_pesos(weights[0].flatten())
    crear_grafica_perdidas(train_losses)

# Crear gráficas de los resultados
def crear_grafica_pesos(pesos):
    img_dir = "Evidencias"
    os.makedirs(img_dir, exist_ok=True)
    plt.plot(pesos, label='Pesos')
    plt.title('Pesos del Modelo')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), shadow=True, ncol=1)
    filename = f"{img_dir}/pesos_modelo.png"
    plt.savefig(filename)
    plt.show()

def crear_grafica_perdidas(train_losses):
    img_dir = "Evidencias"
    os.makedirs(img_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b', label='train_loss')
    plt.title('Pérdida de Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    filename = f"{img_dir}/perdida_entrenamiento.png"
    plt.savefig(filename)
    plt.show()

# Mostrar la interfaz gráfica
def mostrar_ventana():
    global ventana, t_aprendizaje, epocas, treeview
    ventana = Tk()
    ventana.title("Ingrese valores")
    
    Label(ventana, text="Tasa de aprendizaje").grid(row=1, column=0)
    Label(ventana, text="Epocas").grid(row=2, column=0)

    t_aprendizaje = Entry(ventana)
    epocas = Entry(ventana)

    t_aprendizaje.grid(row=1, column=1)
    epocas.grid(row=2, column=1)

    Button(ventana, text="Aceptar", command=principal).grid(row=3, column=0, columnspan=3)

    treeview = ttk.Treeview(ventana, columns=("Error", "Pesos", "Sesgo"), show="headings")
    treeview.heading("Error", text="Error")
    treeview.heading("Pesos", text="Pesos")
    treeview.heading("Sesgo", text="Sesgo")
    treeview.grid(row=4, column=0, columnspan=3)

    ventana.mainloop()

mostrar_ventana()
