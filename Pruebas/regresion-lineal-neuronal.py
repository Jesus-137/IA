import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generación de datos de ejemplo
np.random.seed(42)
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (1000, 1))

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(self.z1, 0)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        return self.forward(X)

# Crear y entrenar la red neuronal
nn = NeuralNetwork(input_size=1, hidden_size=10, output_size=1)
nn.train(X_train_scaled, y_train_scaled, epochs=1000, learning_rate=0.01)

# Hacer predicciones
y_pred_scaled = nn.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calcular el error cuadrático medio (MSE)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Error cuadrático medio: {mse}")

# Graficar los resultados
plt.scatter(X_test, y_test, label='Datos de prueba')
plt.plot(X_test, y_pred, color='red', label='Predicción')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal con Red Neuronal')
plt.show()
