import numpy as np
import matplotlib.pyplot as plt

# Generación de datos de ejemplo
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Función para calcular el error cuadrático medio (MSE)
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# Función de fitness (el inverso del MSE)
def fitness(chromosome):
    m, b = chromosome
    y_pred = m * X + b
    return 1 / mse(y, y_pred)

# Función para crear una población inicial
def create_population(size):
    return np.random.uniform(-10, 10, (size, 2))

# Función de selección (torneo)
def selection(population, fitnesses, k=3):
    selected = np.zeros((len(population), 2))
    for i in range(len(population)):
        candidates = np.random.choice(len(population), k)
        selected[i] = population[candidates[np.argmax(fitnesses[candidates])]]
    return selected

# Función de cruce
def crossover(parent1, parent2):
    child = (parent1 + parent2) / 2
    return child

# Función de mutación
def mutate(chromosome, mutation_rate=0.1, mutation_scale=0.1):
    if np.random.random() < mutation_rate:
        chromosome += np.random.normal(0, mutation_scale, 2)
    return chromosome

# Algoritmo genético principal
def genetic_algorithm(generations=100, population_size=50):
    population = create_population(population_size)
    
    for _ in range(generations):
        fitnesses = np.array([fitness(chromosome) for chromosome in population])
        
        # Selección
        parents = selection(population, fitnesses)
        
        # Cruce y mutación
        new_population = np.zeros_like(population)
        for i in range(0, population_size, 2):
            child1 = crossover(parents[i], parents[i+1])
            child2 = crossover(parents[i], parents[i+1])
            new_population[i] = mutate(child1)
            new_population[i+1] = mutate(child2)
        
        population = new_population
    
    # Seleccionar el mejor cromosoma
    fitnesses = np.array([fitness(chromosome) for chromosome in population])
    best_chromosome = population[np.argmax(fitnesses)]
    
    return best_chromosome

# Ejecutar el algoritmo genético
best_solution = genetic_algorithm()

# Graficar los resultados
plt.scatter(X, y, label='Datos')
plt.plot(X, best_solution[0] * X + best_solution[1], color='red', label='Regresión')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión Lineal con Algoritmo Genético')
plt.show()

print(f"Mejor solución encontrada: y = {best_solution[0]:.4f}x + {best_solution[1]:.4f}")
