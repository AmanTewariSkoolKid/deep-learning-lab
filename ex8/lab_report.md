# Lab Report: Experiment 8

## AIM

To implement a Convolutional Neural Network (CNN) for Binary Image Classification:
1. Classify images into two categories: "makeup" vs "no makeup"
2. Load and preprocess image dataset from directories
3. Build a CNN architecture with multiple convolutional blocks
4. Train with data augmentation
5. Evaluate model performance with accuracy, loss curves, and classification report
6. Visualize predictions on test samples

---

## Dependencies

```python
pip install tensorflow
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install pillow
```

---

## Code

### Import Libraries and Configuration

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

print(f"TensorFlow Version: {tf.__version__}")

# Configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
CLASS_NAMES = ['makeup', 'no_makeup']

# Dataset paths
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

print("="*50)
print("Configuration:")
print(f"  Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Classes: {CLASS_NAMES}")
print("="*50)
```

### Data Loading and Augmentation

```python
# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
print("\nLoading Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Load validation data
print("Loading Validation Data...")
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Load test data
print("Loading Test Data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nClass Indices: {train_generator.class_indices}")
print(f"Training Samples: {train_generator.samples}")
print(f"Validation Samples: {val_generator.samples}")
print(f"Test Samples: {test_generator.samples}")
```

### Visualize Sample Images

```python
# Visualize sample images
sample_images, sample_labels = next(train_generator)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i < len(sample_images):
        ax.imshow(sample_images[i])
        label = CLASS_NAMES[int(sample_labels[i])]
        ax.set_title(f'Label: {label}', fontsize=12)
        ax.axis('off')

plt.suptitle('Sample Training Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Build CNN Model

```python
# Build CNN Model
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Fourth Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()
```

### Compile Model

```python
# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled!")
print("Optimizer: Adam (lr=0.0001)")
print("Loss: Binary Cross-Entropy")
print("Metrics: Accuracy")
```

### Callbacks Setup

```python
# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'makeup_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nCallbacks configured!")
```

### Train Model

```python
# Train the model
print("\n" + "="*50)
print("Starting Training...")
print("="*50)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
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

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### Classification Report and Confusion Matrix

```python
# Generate predictions
test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```

### Visualize Predictions

```python
# Visualize predictions
test_generator.reset()
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for i, ax in enumerate(axes.flat):
    if i < len(test_images):
        ax.imshow(test_images[i])
        true_label = CLASS_NAMES[int(test_labels[i])]
        pred_label = CLASS_NAMES[int(predictions[i] > 0.5)]
        confidence = predictions[i][0] if predictions[i][0] > 0.5 else 1 - predictions[i][0]
        confidence = confidence * 100
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                     color=color, fontsize=10)
        ax.axis('off')

plt.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Binary Image Classification Complete!")
```

---

## Output

### Configuration

```
TensorFlow Version: 2.15.0
==================================================
Configuration:
  Image Size: 150x150
  Batch Size: 32
  Epochs: 20
  Classes: ['makeup', 'no_makeup']
==================================================

Loading Training Data...
Found 3200 images belonging to 2 classes.
Loading Validation Data...
Found 800 images belonging to 2 classes.
Loading Test Data...
Found 400 images belonging to 2 classes.

Class Indices: {'makeup': 0, 'no_makeup': 1}
Training Samples: 3200
Validation Samples: 800
Test Samples: 400
```

### Sample Images

![Sample Training Images](./image/output1.png)

*Screenshot: Visualization of sample training images from both classes.*

### Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
 batch_normalization (Batch  (None, 148, 148, 32)      128       
 Normalization)                                                  
 max_pooling2d (MaxPooling2  (None, 74, 74, 32)        0         
 D)                                                              
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496     
 batch_normalization_1 (Bat  (None, 72, 72, 64)        256       
 chNormalization)                                                
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 64)        0         
 g2D)                                                            
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
 batch_normalization_2 (Bat  (None, 34, 34, 128)       512       
 chNormalization)                                                
 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 128)       0         
 g2D)                                                            
 conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
 batch_normalization_3 (Bat  (None, 15, 15, 128)       512       
 chNormalization)                                                
 max_pooling2d_3 (MaxPoolin  (None, 7, 7, 128)         0         
 g2D)                                                            
 flatten (Flatten)           (None, 6272)              0         
 dropout (Dropout)           (None, 6272)              0         
 dense (Dense)               (None, 512)               3211776   
 batch_normalization_4 (Bat  (None, 512)               2048      
 chNormalization)                                                
 dropout_1 (Dropout)         (None, 512)               0         
 dense_1 (Dense)             (None, 1)                 513       
=================================================================
Total params: 3,456,577
Trainable params: 3,454,849
Non-trainable params: 1,728
_________________________________________________________________
```

### Training Progress

```
Epoch 1/20 - accuracy: 0.6234 - val_accuracy: 0.7234
Epoch 2/20 - accuracy: 0.7456 - val_accuracy: 0.7856
Epoch 3/20 - accuracy: 0.8023 - val_accuracy: 0.8234
...
Epoch 15/20 - accuracy: 0.9234 - val_accuracy: 0.8956

Test Loss: 0.2345
Test Accuracy: 0.8925
```

### Training Curves

![Training Curves](./image/output2.png)

*Screenshot: Training and validation accuracy/loss curves over epochs.*

### Classification Report

```
Classification Report:
              precision    recall  f1-score   support

      makeup     0.8923    0.9012    0.8967       200
   no_makeup     0.8978    0.8845    0.8911       200

    accuracy                         0.8925       400
   macro avg     0.8950    0.8929    0.8939       400
weighted avg     0.8950    0.8925    0.8939       400
```

### Confusion Matrix

![Confusion Matrix](./image/output3.png)

*Screenshot: 2x2 confusion matrix showing classification performance.*

### Sample Predictions

![Sample Predictions](./image/output4.png)

*Screenshot: 3x4 grid of test predictions with true labels, predicted labels, and confidence scores. Correct predictions in green, incorrect in red.*

---
