# CNN MNIST Tutorial - Exercise 7

Complete CNN tutorial for handwritten digit classification using the MNIST dataset.

## Files
- **`cnn_tutorial.ipynb`**: Complete Jupyter notebook with step-by-step CNN implementation
- **`notes.md`**: Comprehensive theory notes, architecture explanation, and resource links
- **`requirements.txt`**: Python dependencies
- **`setup.bat`**: Automated setup script for Windows

## Quick Start

### Option 1: Automated Setup (Windows)
```bash
setup.bat
```

### Option 2: Manual Setup
```bash
# Activate virtual environment (if using shared venv)
..\..\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook cnn_tutorial.ipynb
```

## What's Included

### Notebook Sections:
1. Import libraries
2. Load and explore MNIST dataset
3. Data preprocessing (normalize, reshape, one-hot encode)
4. Build CNN model (Conv2D, MaxPooling, Dropout, Dense layers)
5. Compile model
6. Setup learning rate reduction callback
7. Train model (30 epochs)
8. Visualize training history (accuracy & loss curves)
9. Evaluate on test set
10. Confusion matrix and classification report
11. Visualize correct/incorrect predictions
12. Save model
13. Make predictions on custom input

### Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (0-9)
- **Automatically downloaded** by Keras on first run
- No manual download required

### Expected Performance
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~99%
- **Training Time**: ~5-10 minutes (CPU), ~1-2 minutes (GPU)

## Architecture Overview
```
Conv2D(32, 5x5) → Conv2D(32, 5x5) → MaxPool(2x2) → Dropout(0.25)
  ↓
Conv2D(64, 3x3) → Conv2D(64, 3x3) → MaxPool(2x2) → Dropout(0.25)
  ↓
Flatten → Dense(256) → Dropout(0.5) → Dense(10, softmax)
```

## Resources
- **Original Kaggle Tutorial**: https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial
- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **TensorFlow Docs**: https://www.tensorflow.org/api_docs/python/tf/keras

## Notes
See `notes.md` for detailed explanations of:
- CNN components (Conv2D, pooling, dropout)
- Training configuration
- Evaluation metrics
- Tips for improvement
- Common issues and solutions

---

**Level**: Beginner to Intermediate  
**Time**: 30-45 minutes  
**Prerequisites**: Python, basic deep learning knowledge
