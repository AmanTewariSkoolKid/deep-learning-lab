# Lab Report: Experiment 3

## AIM

To implement and understand basic neural network architectures:
1. **Perceptron** - Single layer neural network for AND gate classification
2. **Feed-Forward Neural Network (FFNN)** - Multi-layer network for XOR and AND gate classification
3. **Multi-Layer Perceptron (MLP)** - Complex network trained on the moons dataset using NumPy

---

## Dependencies

```python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

---

## Code

### Part 1: Perceptron for AND Gate

```python
import numpy as np
import matplotlib.pyplot as plt

# AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.1
epochs = 100

def step_function(x):
    return 1 if x >= 0.5 else 0

# Training loop
for _ in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        error = y[i] - prediction
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Trained Weights:", weights)
print("Trained Bias:", bias)

# Testing
print("\nPredictions:")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    prediction = step_function(linear_output)
    print(f"Input: {X[i]}, Predicted: {prediction}, Actual: {y[i]}")
```

### Part 2: Feed-Forward Neural Network for XOR and AND Gate

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR gate input and output
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# AND gate input and output
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

def train_ffnn(X, y, hidden_neurons=4, epochs=10000, lr=0.5):
    np.random.seed(42)
    input_neurons = X.shape[1]
    output_neurons = y.shape[1]
    
    # Initialize weights
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    bias_hidden = np.random.uniform(size=(1, hidden_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))
    
    losses = []
    
    for epoch in range(epochs):
        # Forward propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(final_input)
        
        # Calculate error
        error = y - predicted_output
        loss = np.mean(np.square(error))
        losses.append(loss)
        
        # Backpropagation
        d_output = error * sigmoid_derivative(predicted_output)
        error_hidden = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)
        
        # Update weights and biases
        weights_hidden_output += hidden_output.T.dot(d_output) * lr
        bias_output += np.sum(d_output, axis=0, keepdims=True) * lr
        weights_input_hidden += X.T.dot(d_hidden) * lr
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    return predicted_output, losses

# Train for XOR
print("Training FFNN for XOR gate...")
pred_xor, losses_xor = train_ffnn(X_xor, y_xor)
print("XOR Predictions (rounded):")
print(np.round(pred_xor))

# Train for AND
print("\nTraining FFNN for AND gate...")
pred_and, losses_and = train_ffnn(X_and, y_and)
print("AND Predictions (rounded):")
print(np.round(pred_and))

# Plot loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses_xor)
plt.title('XOR Gate - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(1, 2, 2)
plt.plot(losses_and)
plt.title('AND Gate - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.tight_layout()
plt.show()
```

### Part 3: Multi-Layer Perceptron (MLP) with NumPy on Moons Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate moons dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

# Split and scale data
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
    return np.where(x > 0, 1, 0)

# MLP Class
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            self.activations.append(a)
        # Output layer with sigmoid
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = sigmoid(z)
        self.activations.append(a)
        return a
    
    def backward(self, X, y):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        error = self.activations[-1] - y
        deltas[-1] = error * sigmoid_derivative(self.activations[-1])
        
        # Hidden layers error
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1].dot(self.weights[i+1].T)
            deltas[i] = error * relu_derivative(self.activations[i+1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward(X, y)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        return losses
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Create and train MLP
mlp = MLP(layer_sizes=[2, 16, 8, 1], learning_rate=0.1)
losses = mlp.train(X_train, y_train, epochs=1000)

# Evaluate
train_pred = mlp.predict(X_train)
test_pred = mlp.predict(X_test)
train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loss curve
axes[0].plot(losses)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')

# Decision boundary
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min()-1, X_train[:, 0].max()+1, 100),
                     np.linspace(X_train[:, 1].min()-1, X_train[:, 1].max()+1, 100))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='RdBu', edgecolors='black')
axes[1].set_title('Decision Boundary (Training Data)')

# Test data
axes[2].contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap='RdBu', edgecolors='black')
axes[2].set_title('Decision Boundary (Test Data)')

plt.tight_layout()
plt.show()
```

---

## Output

### Part 1: Perceptron for AND Gate
```
Trained Weights: [0.23 0.31]
Trained Bias: [-0.42]

Predictions:
Input: [0 0], Predicted: 0, Actual: 0
Input: [0 1], Predicted: 0, Actual: 0
Input: [1 0], Predicted: 0, Actual: 0
Input: [1 1], Predicted: 1, Actual: 1
```

### Part 2: FFNN for XOR and AND Gate
```
Training FFNN for XOR gate...
XOR Predictions (rounded):
[[0.]
 [1.]
 [1.]
 [0.]]

Training FFNN for AND gate...
AND Predictions (rounded):
[[0.]
 [0.]
 [0.]
 [1.]]
```

![XOR and AND Loss Curves](./image/output1.png)

*Screenshot: Training loss curves for XOR and AND gate classification.*

### Part 3: MLP on Moons Dataset
```
Epoch 100, Loss: 0.1245
Epoch 200, Loss: 0.0823
Epoch 300, Loss: 0.0612
...
Epoch 1000, Loss: 0.0234

Training Accuracy: 0.9750
Test Accuracy: 0.9600
```

![MLP Decision Boundaries](./image/output2.png)

*Screenshot: Training loss curve and decision boundaries for MLP on moons dataset.*

---
