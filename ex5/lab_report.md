# Lab Report: Experiment 5

## AIM

To implement Transfer Learning for Image Classification using pre-trained models:
1. Load and preprocess the Flower Photos dataset (5 classes: daisy, dandelion, roses, sunflowers, tulips)
2. Apply data augmentation techniques
3. Use MobileNetV2 as a pre-trained feature extractor
4. Fine-tune the model for flower classification
5. Evaluate model performance and visualize training metrics

---

## Dependencies

```python
pip install tensorflow
pip install matplotlib
pip install numpy
pip install pillow
```

---

## Code

### Configuration (config_transfer.json)

```json
{
    "image_size": [224, 224],
    "batch_size": 32,
    "epochs": 10,
    "fine_tune_epochs": 5,
    "fine_tune_at": 100,
    "learning_rate": 0.0001,
    "fine_tune_lr": 0.00001,
    "num_classes": 5,
    "validation_split": 0.2,
    "data_dir": "data/flower_photos",
    "model_save_path": "models/flower_classifier.h5"
}
```

### Main Training Script

```python
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Load configuration
with open('config_transfer.json', 'r') as f:
    config = json.load(f)

IMG_SIZE = tuple(config['image_size'])
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
FINE_TUNE_EPOCHS = config['fine_tune_epochs']
FINE_TUNE_AT = config['fine_tune_at']
LEARNING_RATE = config['learning_rate']
FINE_TUNE_LR = config['fine_tune_lr']
NUM_CLASSES = config['num_classes']
VAL_SPLIT = config['validation_split']
DATA_DIR = config['data_dir']

print("="*50)
print("Configuration Loaded:")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print("="*50)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VAL_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT
)

# Load training data
print("\nLoading Training Data...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("Loading Validation Data...")
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nClass Indices: {train_generator.class_indices}")
print(f"Training Samples: {train_generator.samples}")
print(f"Validation Samples: {val_generator.samples}")

# Build Model with Transfer Learning
print("\n" + "="*50)
print("Building Model with MobileNetV2 Backbone...")
print("="*50)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Build classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Initial Training (Feature Extraction)
print("\n" + "="*50)
print("Phase 1: Feature Extraction Training")
print("="*50)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

# Fine-tuning
print("\n" + "="*50)
print("Phase 2: Fine-Tuning")
print("="*50)

# Unfreeze layers for fine-tuning
base_model.trainable = True

# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"Fine-tuning from layer {FINE_TUNE_AT}")
print(f"Total layers: {len(base_model.layers)}")
print(f"Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
total_epochs = EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_generator,
    verbose=1
)

# Merge histories
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Evaluation
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

val_loss_final, val_acc_final = model.evaluate(val_generator, verbose=1)
print(f"\nFinal Validation Accuracy: {val_acc_final:.4f}")
print(f"Final Validation Loss: {val_loss_final:.4f}")

# Plot Training History
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', linewidth=2)
plt.plot(val_acc, label='Validation Accuracy', linewidth=2)
plt.axvline(x=EPOCHS-1, color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', linewidth=2)
plt.plot(val_loss, label='Validation Loss', linewidth=2)
plt.axvline(x=EPOCHS-1, color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save model
model.save(config['model_save_path'])
print(f"\nModel saved to {config['model_save_path']}")

# Sample Predictions
print("\n" + "="*50)
print("Sample Predictions")
print("="*50)

class_names = list(train_generator.class_indices.keys())

# Get a batch of validation images
val_images, val_labels = next(val_generator)
predictions = model.predict(val_images)

# Display predictions
plt.figure(figsize=(15, 10))
for i in range(min(9, len(val_images))):
    plt.subplot(3, 3, i+1)
    plt.imshow(val_images[i])
    true_label = class_names[np.argmax(val_labels[i])]
    pred_label = class_names[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
              color=color, fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\n✓ Transfer Learning Complete!")
```

---

## Output

### Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_224 (Func  (None, 7, 7, 1280)       2257984   
 tional)                                                         
                                                                 
 global_average_pooling2d (  (None, 1280)             0         
 GlobalAveragePooling2D)                                         
                                                                 
 dropout (Dropout)           (None, 1280)             0         
                                                                 
 dense (Dense)               (None, 256)              327936    
                                                                 
 dropout_1 (Dropout)         (None, 256)              0         
                                                                 
 dense_1 (Dense)             (None, 5)                1285      
                                                                 
=================================================================
Total params: 2,587,205
Trainable params: 329,221
Non-trainable params: 2,257,984
_________________________________________________________________
```

### Training Progress

```
Phase 1: Feature Extraction Training
Epoch 1/10 - accuracy: 0.6234 - val_accuracy: 0.8512
Epoch 2/10 - accuracy: 0.8023 - val_accuracy: 0.8734
...
Epoch 10/10 - accuracy: 0.8945 - val_accuracy: 0.9023

Phase 2: Fine-Tuning
Epoch 11/15 - accuracy: 0.9123 - val_accuracy: 0.9234
Epoch 12/15 - accuracy: 0.9245 - val_accuracy: 0.9312
...
Epoch 15/15 - accuracy: 0.9456 - val_accuracy: 0.9423

Final Validation Accuracy: 0.9423
Final Validation Loss: 0.1876
```

### Training Curves

![Training Curves](./image/output1.png)

*Screenshot: Training and validation accuracy/loss curves showing the improvement after fine-tuning starts at epoch 10.*

### Sample Predictions

![Sample Predictions](./image/output2.png)

*Screenshot: 3x3 grid of sample flower predictions with true labels, predicted labels, and confidence scores. Correct predictions shown in green, incorrect in red.*

---
