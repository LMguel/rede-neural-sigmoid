import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Função de ativação Sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Criando dataset "moons"
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Transformando y em formato coluna

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando pesos e bias
np.random.seed(42)
weights = np.random.randn(2, 1)
bias = np.random.randn()

# Hiperparâmetros
learning_rate = 0.1
epochs = 10000

# Treinamento da rede
errors = []
for epoch in range(epochs):
    # Forward Pass
    weighted_sum = np.dot(X_train, weights) + bias
    output = sigmoid(weighted_sum)
    
    # Cálculo do erro
    error = y_train - output
    errors.append(np.mean(np.square(error)))

    # Backpropagation
    gradient = error * sigmoid_derivative(weighted_sum)
    
    # Atualização dos pesos
    weights += learning_rate * np.dot(X_train.T, gradient)
    bias += learning_rate * np.sum(gradient)

# Teste do modelo
predictions = sigmoid(np.dot(X_test, weights) + bias)
predictions = (predictions > 0.5).astype(int)

# Acurácia
accuracy = np.mean(predictions == y_test) * 100
print(f'Acurácia com ativação Sigmoid: {accuracy:.2f}%')

# Gráfico do erro ao longo do treinamento
plt.plot(errors)
plt.title("Erro durante o treinamento (Sigmoid)")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()
