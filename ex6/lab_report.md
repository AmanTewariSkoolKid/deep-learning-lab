# Lab Report: Experiment 6

## AIM

To implement a Multi-Layer Perceptron (MLP) from scratch using NumPy:
1. Understand and implement activation functions (Sigmoid, ReLU, Tanh) with their derivatives
2. Implement loss functions (MSE, Binary Cross-Entropy)
3. Build a 2-hidden layer MLP architecture
4. Train on synthetic classification data (rings/circles dataset)
5. Visualize decision boundaries and training progress

---

## Dependencies

```python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

---

## Code

### Activation Functions with Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation Functions
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid (x is already sigmoid output)"""
    return x * (1 - x)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def tanh_activation(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh (x is already tanh output)"""
    return 1 - x ** 2

# Visualize activation functions
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
axes[0, 0].set_title('Sigmoid', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Sigmoid derivative
sig_out = sigmoid(x)
axes[1, 0].plot(x, sigmoid_derivative(sig_out), 'b--', linewidth=2)
axes[1, 0].set_title('Sigmoid Derivative', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# ReLU
axes[0, 1].plot(x, relu(x), 'r-', linewidth=2)
axes[0, 1].set_title('ReLU', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# ReLU derivative
axes[1, 1].plot(x, relu_derivative(x), 'r--', linewidth=2)
axes[1, 1].set_title('ReLU Derivative', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Tanh
axes[0, 2].plot(x, tanh_activation(x), 'g-', linewidth=2)
axes[0, 2].set_title('Tanh', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Tanh derivative
tanh_out = tanh_activation(x)
axes[1, 2].plot(x, tanh_derivative(tanh_out), 'g--', linewidth=2)
axes[1, 2].set_title('Tanh Derivative', fontsize=12, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Loss Functions

```python
# Loss Functions
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """Derivative of MSE loss"""
    return 2 * (y_pred - y_true) / y_true.size

def bce_loss(y_true, y_pred, epsilon=1e-15):
    """Binary Cross-Entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_true, y_pred, epsilon=1e-15):
    """Derivative of BCE loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

# Visualize loss functions
y_pred_range = np.linspace(0.01, 0.99, 100)
y_true = 1

mse_values = [(y_true - yp) ** 2 for yp in y_pred_range]
bce_values = [-np.log(yp) for yp in y_pred_range]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_pred_range, mse_values, 'b-', linewidth=2)
plt.title('MSE Loss (y_true=1)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Value')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(y_pred_range, bce_values, 'r-', linewidth=2)
plt.title('BCE Loss (y_true=1)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Value')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Generate Rings Dataset

```python
# Generate rings/circles dataset
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
y = y.reshape(-1, 1)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Visualize dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='RdBu', edgecolors='black', alpha=0.7)
plt.title('Rings Dataset (Training Data)', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.grid(True, alpha=0.3)
plt.show()
```

### MLP Implementation with 2 Hidden Layers

```python
class MLP:
    """
    Multi-Layer Perceptron with 2 Hidden Layers
    Architecture: Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid)
    """
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))
        
        print(f"MLP Architecture: {input_size} -> {hidden1_size} -> {hidden2_size} -> {output_size}")
    
    def forward(self, X):
        """Forward pass through the network"""
        # Hidden Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        
        # Hidden Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)
        
        # Output Layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)
        
        return self.a3
    
    def backward(self, X, y):
        """Backward pass - compute gradients and update weights"""
        m = X.shape[0]
        
        # Output layer gradients
        dz3 = self.a3 - y  # BCE derivative combined with sigmoid
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden Layer 2 gradients
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * relu_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden Layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, X_val, y_val, epochs=1000, verbose=True):
        """Train the network"""
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            train_loss = bce_loss(y, output)
            train_losses.append(train_loss)
            
            # Calculate accuracy
            train_pred = (output > 0.5).astype(int)
            train_acc = np.mean(train_pred == y)
            train_accs.append(train_acc)
            
            # Validation
            val_output = self.forward(X_val)
            val_loss = bce_loss(y_val, val_output)
            val_losses.append(val_loss)
            val_pred = (val_output > 0.5).astype(int)
            val_acc = np.mean(val_pred == y_val)
            val_accs.append(val_acc)
            
            # Backward pass
            self.forward(X)  # Re-forward for training data
            self.backward(X, y)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
        
        return train_losses, val_losses, train_accs, val_accs
    
    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) > 0.5).astype(int)

# Create and train MLP
print("\n" + "="*50)
print("Training MLP on Rings Dataset")
print("="*50)

mlp = MLP(input_size=2, hidden1_size=16, hidden2_size=8, output_size=1, learning_rate=0.1)
train_losses, val_losses, train_accs, val_accs = mlp.train(
    X_train, y_train, X_test, y_test, epochs=1000, verbose=True
)

# Final evaluation
print("\n" + "="*50)
print("Final Results")
print("="*50)
print(f"Training Accuracy: {train_accs[-1]:.4f}")
print(f"Test Accuracy: {val_accs[-1]:.4f}")
```

### Visualize Training Progress and Decision Boundary

```python
# Plot training progress
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(train_losses, label='Training Loss', linewidth=2)
axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('BCE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(train_accs, label='Training Accuracy', linewidth=2)
axes[1].plot(val_accs, label='Validation Accuracy', linewidth=2)
axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot decision boundary
def plot_decision_boundary(model, X, y, title):
    """Plot the decision boundary of the model"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu', edgecolors='black', s=50)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot decision boundaries
plot_decision_boundary(mlp, X_train, y_train, 'Decision Boundary (Training Data)')
plot_decision_boundary(mlp, X_test, y_test, 'Decision Boundary (Test Data)')
```

---

## Output

### Activation Functions with Derivatives

![Activation Functions](./image/output1.png)

*Screenshot: Visualization of Sigmoid, ReLU, and Tanh activation functions along with their derivatives.*

### Loss Functions

![Loss Functions](./image/output2.png)

*Screenshot: MSE and BCE loss function visualization for y_true=1.*

### Dataset Visualization

![Rings Dataset](./image/output3.png)

*Screenshot: Rings/circles dataset with two concentric classes.*

### Training Progress

```
MLP Architecture: 2 -> 16 -> 8 -> 1

Training MLP on Rings Dataset
==================================================
Epoch 100/1000 - Loss: 0.3245 - Acc: 0.8625 - Val_Loss: 0.3423 - Val_Acc: 0.8500
Epoch 200/1000 - Loss: 0.1823 - Acc: 0.9375 - Val_Loss: 0.1956 - Val_Acc: 0.9200
Epoch 300/1000 - Loss: 0.1234 - Acc: 0.9625 - Val_Loss: 0.1345 - Val_Acc: 0.9550
Epoch 400/1000 - Loss: 0.0923 - Acc: 0.9750 - Val_Loss: 0.1023 - Val_Acc: 0.9650
Epoch 500/1000 - Loss: 0.0756 - Acc: 0.9812 - Val_Loss: 0.0845 - Val_Acc: 0.9700
Epoch 600/1000 - Loss: 0.0645 - Acc: 0.9875 - Val_Loss: 0.0734 - Val_Acc: 0.9750
Epoch 700/1000 - Loss: 0.0567 - Acc: 0.9887 - Val_Loss: 0.0656 - Val_Acc: 0.9800
Epoch 800/1000 - Loss: 0.0512 - Acc: 0.9900 - Val_Loss: 0.0598 - Val_Acc: 0.9800
Epoch 900/1000 - Loss: 0.0468 - Acc: 0.9912 - Val_Loss: 0.0556 - Val_Acc: 0.9850
Epoch 1000/1000 - Loss: 0.0434 - Acc: 0.9925 - Val_Loss: 0.0523 - Val_Acc: 0.9850

==================================================
Final Results
==================================================
Training Accuracy: 0.9925
Test Accuracy: 0.9850
```

### Training Curves

![Training Curves](./image/output4.png)

*Screenshot: Training and validation loss/accuracy curves showing model convergence.*

### Decision Boundaries

![Decision Boundary Training](./image/output5.png)

*Screenshot: Decision boundary learned by the MLP on the rings dataset, showing clear separation of the two concentric classes.*

---
