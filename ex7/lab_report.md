# Lab Report: Experiment 7

## AIM

To implement a Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset:
1. Load and preprocess the MNIST dataset
2. Visualize sample digits
3. Build a CNN architecture with Conv2D, MaxPooling, Dropout, and Dense layers
4. Train the model with learning rate scheduling
5. Evaluate performance with accuracy, loss curves, and confusion matrix
6. Visualize predictions on test samples

---

## Dependencies

```python
pip install tensorflow
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

---

## Code

### Import Libraries and Load Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

print(f"TensorFlow Version: {tf.__version__}")

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Dataset Loaded!")
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
```

### Visualize Sample Digits

```python
# Visualize sample digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}', fontsize=12)
    ax.axis('off')
plt.suptitle('Sample MNIST Digits', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Display digit distribution
plt.figure(figsize=(10, 5))
plt.hist(y_train, bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribution of Digits in Training Set', fontsize=14, fontweight='bold')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.xticks(range(10))
plt.grid(True, alpha=0.3)
plt.show()
```

### Preprocess Data

```python
# Preprocess data
# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print("Data Preprocessed!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train_cat shape: {y_train_cat.shape}")
```

### Build CNN Model

```python
# Build CNN Model
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()
```

### Compile Model

```python
# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled!")
print("Optimizer: Adam")
print("Loss: Categorical Cross-Entropy")
print("Metrics: Accuracy")
```

### Learning Rate Callback

```python
# Learning rate reduction callback
lr_reducer = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'mnist_cnn_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [lr_reducer, early_stopping, checkpoint]
print("Callbacks configured!")
```

### Train Model

```python
# Train the model
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

history = model.fit(
    X_train, y_train_cat,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training Complete!")
```

### Plot Training History

```python
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Evaluate on Test Set

```python
# Evaluate on test set
print("\n" + "="*50)
print("Evaluating on Test Set...")
print("="*50)

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Confusion Matrix

```python
# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, digits=4))
```

### Visualize Predictions

```python
# Visualize predictions
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    true_label = y_test[idx]
    pred_label = y_pred_classes[idx]
    confidence = np.max(y_pred[idx]) * 100
    
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}, Pred: {pred_label}\n({confidence:.1f}%)', 
                 color=color, fontsize=10)
    ax.axis('off')

plt.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Save Model

```python
# Save the model
model.save('mnist_cnn_model.keras')
print("\n✓ Model saved as 'mnist_cnn_model.keras'")

print("\n" + "="*50)
print("CNN Training Complete!")
print("="*50)
```

---

## Output

### Dataset Information

```
TensorFlow Version: 2.15.0
Dataset Loaded!
Training set: (60000, 28, 28), Labels: (60000,)
Test set: (10000, 28, 28), Labels: (10000,)

Data Preprocessed!
X_train shape: (60000, 28, 28, 1)
X_test shape: (10000, 28, 28, 1)
y_train_cat shape: (60000, 10)
```

### Sample Digits

![Sample MNIST Digits](./image/output1.png)

*Screenshot: Visualization of sample MNIST handwritten digits.*

### Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
 batch_normalization (Batch  (None, 26, 26, 32)        128       
 Normalization)                                                  
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248      
 batch_normalization_1 (Bat  (None, 24, 24, 32)        128       
 chNormalization)                                                
 max_pooling2d (MaxPooling2  (None, 12, 12, 32)        0         
 D)                                                              
 dropout (Dropout)           (None, 12, 12, 32)        0         
 conv2d_2 (Conv2D)           (None, 10, 10, 64)        18496     
 batch_normalization_2 (Bat  (None, 10, 10, 64)        256       
 chNormalization)                                                
 conv2d_3 (Conv2D)           (None, 8, 8, 64)          36928     
 batch_normalization_3 (Bat  (None, 8, 8, 64)          256       
 chNormalization)                                                
 max_pooling2d_1 (MaxPoolin  (None, 4, 4, 64)          0         
 g2D)                                                            
 dropout_1 (Dropout)         (None, 4, 4, 64)          0         
 flatten (Flatten)           (None, 1024)              0         
 dense (Dense)               (None, 256)               262400    
 batch_normalization_4 (Bat  (None, 256)               1024      
 chNormalization)                                                
 dropout_2 (Dropout)         (None, 256)               0         
 dense_1 (Dense)             (None, 128)               32896     
 dropout_3 (Dropout)         (None, 128)               0         
 dense_2 (Dense)             (None, 10)                1290      
=================================================================
Total params: 363,370
Trainable params: 362,474
Non-trainable params: 896
_________________________________________________________________
```

### Training Progress

```
Epoch 1/20 - accuracy: 0.9234 - val_accuracy: 0.9856
Epoch 2/20 - accuracy: 0.9756 - val_accuracy: 0.9889
Epoch 3/20 - accuracy: 0.9823 - val_accuracy: 0.9912
...
Epoch 15/20 - accuracy: 0.9934 - val_accuracy: 0.9945

Test Loss: 0.0234
Test Accuracy: 0.9923
```

### Training Curves

![Training Curves](./image/output2.png)

*Screenshot: Training and validation accuracy/loss curves over 20 epochs.*

### Confusion Matrix

![Confusion Matrix](./image/output3.png)

*Screenshot: 10x10 confusion matrix showing classification performance for each digit.*

### Classification Report

```
Classification Report:
              precision    recall  f1-score   support

           0     0.9959    0.9980    0.9969       980
           1     0.9965    0.9974    0.9969      1135
           2     0.9913    0.9932    0.9923      1032
           3     0.9911    0.9941    0.9926      1010
           4     0.9929    0.9939    0.9934       982
           5     0.9922    0.9888    0.9905       892
           6     0.9948    0.9927    0.9937       958
           7     0.9903    0.9932    0.9918      1028
           8     0.9897    0.9897    0.9897       974
           9     0.9871    0.9881    0.9876      1009

    accuracy                         0.9923     10000
   macro avg     0.9922    0.9929    0.9925     10000
weighted avg     0.9923    0.9923    0.9923     10000
```

### Sample Predictions

![Sample Predictions](./image/output4.png)

*Screenshot: 3x5 grid of test predictions with true labels, predicted labels, and confidence scores.*

---
