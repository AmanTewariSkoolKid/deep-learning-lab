# Lab Report: Experiment 4

## AIM

To understand and implement fundamental deep learning concepts:
1. **Activation Functions** - Visualize sigmoid, tanh, ReLU, Leaky ReLU, and softmax
2. **Loss Functions** - Implement and visualize MSE and Binary Cross-Entropy losses
3. **Backpropagation** - Implement backpropagation algorithm from scratch
4. **Optimizer Comparison** - Compare SGD, Momentum, and Adam optimizers on classification task

---

## Dependencies

```python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

---

## Code

### Part 1: Activation Functions Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Create input range
x = np.linspace(-5, 5, 100)

# Plot all activation functions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
axes[0, 0].set_title('Sigmoid', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('σ(x)')
axes[0, 0].grid(True, alpha=0.3)

# Tanh
axes[0, 1].plot(x, tanh(x), 'g-', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
axes[0, 1].set_title('Tanh', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('tanh(x)')
axes[0, 1].grid(True, alpha=0.3)

# ReLU
axes[0, 2].plot(x, relu(x), 'r-', linewidth=2)
axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
axes[0, 2].axvline(x=0, color='k', linewidth=0.5)
axes[0, 2].set_title('ReLU', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('ReLU(x)')
axes[0, 2].grid(True, alpha=0.3)

# Leaky ReLU
axes[1, 0].plot(x, leaky_relu(x), 'm-', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
axes[1, 0].set_title('Leaky ReLU (α=0.01)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('LeakyReLU(x)')
axes[1, 0].grid(True, alpha=0.3)

# Softmax example
softmax_input = np.array([2.0, 1.0, 0.1])
softmax_output = softmax(softmax_input)
axes[1, 1].bar(['Class 0', 'Class 1', 'Class 2'], softmax_output, color=['red', 'green', 'blue'])
axes[1, 1].set_title('Softmax Output', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_ylim([0, 1])

# All activations comparison
axes[1, 2].plot(x, sigmoid(x), 'b-', label='Sigmoid', linewidth=2)
axes[1, 2].plot(x, tanh(x), 'g-', label='Tanh', linewidth=2)
axes[1, 2].plot(x, relu(x), 'r-', label='ReLU', linewidth=2)
axes[1, 2].plot(x, leaky_relu(x), 'm-', label='Leaky ReLU', linewidth=2)
axes[1, 2].axhline(y=0, color='k', linewidth=0.5)
axes[1, 2].axvline(x=0, color='k', linewidth=0.5)
axes[1, 2].set_title('All Activations Comparison', fontsize=14, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Part 2: Loss Functions Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

# Mean Squared Error
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy
def bce_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Visualize loss functions
y_true = 1  # True label
y_pred_range = np.linspace(0.01, 0.99, 100)

mse_values = [(y_true - y_pred) ** 2 for y_pred in y_pred_range]
bce_values = [-np.log(y_pred) for y_pred in y_pred_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSE
axes[0].plot(y_pred_range, mse_values, 'b-', linewidth=2)
axes[0].set_title('Mean Squared Error (y_true=1)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Value')
axes[0].set_ylabel('MSE Loss')
axes[0].grid(True, alpha=0.3)

# BCE
axes[1].plot(y_pred_range, bce_values, 'r-', linewidth=2)
axes[1].set_title('Binary Cross-Entropy Loss (y_true=1)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Value')
axes[1].set_ylabel('BCE Loss')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Example Loss Calculations:")
print(f"MSE(y_true=1, y_pred=0.9): {mse_loss(1, 0.9):.4f}")
print(f"MSE(y_true=1, y_pred=0.1): {mse_loss(1, 0.1):.4f}")
print(f"BCE(y_true=1, y_pred=0.9): {bce_loss(np.array([1]), np.array([0.9])):.4f}")
print(f"BCE(y_true=1, y_pred=0.1): {bce_loss(np.array([1]), np.array([0.1])):.4f}")
```

### Part 3: Backpropagation Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Neural Network with Backpropagation
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = sigmoid(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, y):
        m = y.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer gradient
        output_error = self.activations[-1] - y
        deltas[-1] = output_error * sigmoid_derivative(self.activations[-1])
        
        # Hidden layers gradient (backpropagation)
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = error * relu_derivative(self.activations[i + 1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            self.backward(y)
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        return losses
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Train network
nn = NeuralNetwork([2, 16, 8, 1], learning_rate=0.1)
losses = nn.train(X_train, y_train, epochs=1000)

# Evaluate
train_acc = np.mean(nn.predict(X_train) == y_train)
test_acc = np.mean(nn.predict(X_test) == y_test)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Backpropagation Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True, alpha=0.3)
plt.show()
```

### Part 4: Optimizer Comparison (SGD vs Momentum vs Adam)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_redundant=2, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class MLPWithOptimizer:
    def __init__(self, layers, optimizer='sgd', lr=0.01, beta1=0.9, beta2=0.999):
        self.layers = layers
        self.optimizer = optimizer
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        
        self.weights = []
        self.biases = []
        self.v_w = []  # Momentum/Adam first moment
        self.v_b = []
        self.s_w = []  # Adam second moment
        self.s_b = []
        self.t = 0  # Time step for Adam
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.v_w.append(np.zeros_like(w))
            self.v_b.append(np.zeros_like(b))
            self.s_w.append(np.zeros_like(w))
            self.s_b.append(np.zeros_like(b))
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = np.maximum(0, z)  # ReLU
            self.activations.append(a)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        output = sigmoid(z)
        self.activations.append(output)
        return output
    
    def backward(self, y):
        m = y.shape[0]
        self.t += 1
        
        deltas = [None] * len(self.weights)
        output_error = self.activations[-1] - y
        deltas[-1] = output_error
        
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = error * (self.activations[i + 1] > 0).astype(float)
        
        for i in range(len(self.weights)):
            dw = self.activations[i].T.dot(deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            if self.optimizer == 'sgd':
                self.weights[i] -= self.lr * dw
                self.biases[i] -= self.lr * db
                
            elif self.optimizer == 'momentum':
                self.v_w[i] = self.beta1 * self.v_w[i] + self.lr * dw
                self.v_b[i] = self.beta1 * self.v_b[i] + self.lr * db
                self.weights[i] -= self.v_w[i]
                self.biases[i] -= self.v_b[i]
                
            elif self.optimizer == 'adam':
                self.v_w[i] = self.beta1 * self.v_w[i] + (1 - self.beta1) * dw
                self.v_b[i] = self.beta1 * self.v_b[i] + (1 - self.beta1) * db
                self.s_w[i] = self.beta2 * self.s_w[i] + (1 - self.beta2) * (dw ** 2)
                self.s_b[i] = self.beta2 * self.s_b[i] + (1 - self.beta2) * (db ** 2)
                
                v_w_corrected = self.v_w[i] / (1 - self.beta1 ** self.t)
                v_b_corrected = self.v_b[i] / (1 - self.beta1 ** self.t)
                s_w_corrected = self.s_w[i] / (1 - self.beta2 ** self.t)
                s_b_corrected = self.s_b[i] / (1 - self.beta2 ** self.t)
                
                self.weights[i] -= self.lr * v_w_corrected / (np.sqrt(s_w_corrected) + self.epsilon)
                self.biases[i] -= self.lr * v_b_corrected / (np.sqrt(s_b_corrected) + self.epsilon)
    
    def train(self, X, y, epochs=500):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            self.backward(y)
        return losses
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Train with different optimizers
print("Training with SGD...")
sgd_model = MLPWithOptimizer([10, 32, 16, 1], optimizer='sgd', lr=0.01)
sgd_losses = sgd_model.train(X_train, y_train, epochs=500)

print("Training with Momentum...")
momentum_model = MLPWithOptimizer([10, 32, 16, 1], optimizer='momentum', lr=0.01)
momentum_losses = momentum_model.train(X_train, y_train, epochs=500)

print("Training with Adam...")
adam_model = MLPWithOptimizer([10, 32, 16, 1], optimizer='adam', lr=0.01)
adam_losses = adam_model.train(X_train, y_train, epochs=500)

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(sgd_losses, label='SGD', linewidth=2)
plt.plot(momentum_losses, label='Momentum', linewidth=2)
plt.plot(adam_losses, label='Adam', linewidth=2)
plt.title('Optimizer Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print accuracies
print("\nTest Accuracies:")
print(f"SGD: {np.mean(sgd_model.predict(X_test) == y_test):.4f}")
print(f"Momentum: {np.mean(momentum_model.predict(X_test) == y_test):.4f}")
print(f"Adam: {np.mean(adam_model.predict(X_test) == y_test):.4f}")
```

---

## Output

### Part 1: Activation Functions

![Activation Functions](./image/output1.png)

*Screenshot: Visualization of sigmoid, tanh, ReLU, Leaky ReLU, and softmax activation functions.*

### Part 2: Loss Functions

```
Example Loss Calculations:
MSE(y_true=1, y_pred=0.9): 0.0100
MSE(y_true=1, y_pred=0.1): 0.8100
BCE(y_true=1, y_pred=0.9): 0.1054
BCE(y_true=1, y_pred=0.1): 2.3026
```

![Loss Functions](./image/output2.png)

*Screenshot: MSE and Binary Cross-Entropy loss visualization.*

### Part 3: Backpropagation

```
Epoch 200, Loss: 0.3245
Epoch 400, Loss: 0.1823
Epoch 600, Loss: 0.1234
Epoch 800, Loss: 0.0987
Epoch 1000, Loss: 0.0823

Training Accuracy: 0.9625
Test Accuracy: 0.9500
```

![Backpropagation Loss](./image/output3.png)

*Screenshot: Training loss curve showing convergence during backpropagation.*

### Part 4: Optimizer Comparison

```
Test Accuracies:
SGD: 0.8450
Momentum: 0.8650
Adam: 0.8750
```

![Optimizer Comparison](./image/output4.png)

*Screenshot: Training loss comparison between SGD, Momentum, and Adam optimizers. Adam converges fastest followed by Momentum and then SGD.*

---
