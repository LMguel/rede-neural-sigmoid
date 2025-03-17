import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ============================
# Funções de ativação
# ============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# ============================
# Criando o dataset
# ============================
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y.reshape(-1, 1)  # Transformando y em formato coluna

# Plot do dataset
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Spectral)
plt.title("Dataset: Lua Crescente")
plt.show()

# Hiperparâmetros
learning_rate = 0.1
epochs = 50000

# ============================
# Treinamento com SIGMOID
# ============================
np.random.seed(42)
weights_sigmoid = np.random.randn(2, 1)
bias_sigmoid = np.random.randn()

errors_sigmoid = []
for epoch in range(epochs):
    # Forward pass
    weighted_sum = np.dot(X, weights_sigmoid) + bias_sigmoid
    output = sigmoid(weighted_sum)

    # Cálculo do erro
    error = y - output
    errors_sigmoid.append(np.mean(np.square(error)))

    # Backpropagation
    d_error = -2 * error / len(X)
    d_output = sigmoid_derivative(weighted_sum)
    gradient = d_error * d_output

    # Atualização dos pesos e bias
    weights_sigmoid -= learning_rate * np.dot(X.T, gradient)
    bias_sigmoid -= learning_rate * np.sum(gradient)

# ============================
# Treinamento com ReLU
# ============================
np.random.seed(42)
weights_relu = np.random.randn(2, 1)
bias_relu = np.random.randn()

errors_relu = []
for epoch in range(epochs):
    # Forward pass
    weighted_sum = np.dot(X, weights_relu) + bias_relu
    output = relu(weighted_sum)

    # Cálculo do erro
    error = y - output
    errors_relu.append(np.mean(np.square(error)))

    # Backpropagation
    d_error = -2 * error / len(X)
    d_output = relu_derivative(weighted_sum)
    gradient = d_error * d_output

    # Atualização dos pesos e bias
    weights_relu -= learning_rate * np.dot(X.T, gradient)
    bias_relu -= learning_rate * np.sum(gradient)

# ============================
# Comparação dos resultados
# ============================
plt.plot(range(epochs), errors_sigmoid, label="Sigmoid", color="blue")
plt.plot(range(epochs), errors_relu, label="ReLU", color="red")
plt.title("Erro ao longo do treinamento")
plt.xlabel("Época")
plt.ylabel("Erro")
plt.legend()
plt.show()

# Teste da rede com Sigmoid
print("\nTeste da rede com Sigmoid:")
predictions_sigmoid = sigmoid(np.dot(X, weights_sigmoid) + bias_sigmoid)
predictions_sigmoid = (predictions_sigmoid > 0.5).astype(int)
print(f"Acurácia (Sigmoid): {np.mean(predictions_sigmoid == y) * 100:.2f}%")

# Teste da rede com ReLU
print("\nTeste da rede com ReLU:")
predictions_relu = relu(np.dot(X, weights_relu) + bias_relu)
predictions_relu = (predictions_relu > 0.5).astype(int)
print(f"Acurácia (ReLU): {np.mean(predictions_relu == y) * 100:.2f}%")
