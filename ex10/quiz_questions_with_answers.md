# Object Detection using R-CNN Quiz - Answer Key

## **Multiple Choice Questions**

**1. What is the primary purpose of U-Net architecture?**
- A) Image classification
- **B) Semantic segmentation** ✓
- C) Object detection
- D) Image generation

**2. What does IoU stand for in the context of segmentation?**
- A) Image over Union
- **B) Intersection over Union** ✓
- C) Input over Update
- D) Index of Uncertainty

**3. What is the range of values for the Dice Score?**
- A) -1 to 1
- B) 0 to 100
- **C) 0 to 1** ✓
- D) 1 to 10

**4. In the U-Net architecture, what is the purpose of skip connections?**
- A) To speed up training
- B) To reduce memory usage
- **C) To recover spatial information lost during downsampling** ✓
- D) To prevent gradient vanishing

**5. What activation function is used in the output layer for binary segmentation?**
- A) ReLU
- B) Softmax
- **C) Sigmoid** ✓
- D) Tanh

**6. The encoder path in U-Net is also called:**
- A) Expansive path
- **B) Contracting path** ✓
- C) Skip path
- D) Bottleneck path

**7. What does the binarization threshold of 0.5 mean for masks?**
- A) Pixels below 0.5 become 1, above become 0
- **B) Pixels above 0.5 become 1, below become 0** ✓
- C) All pixels become 0.5
- D) Pixels are normalized to 0.5

**8. Which callback prevents overfitting by stopping training early?**
- A) ModelCheckpoint
- **B) EarlyStopping** ✓
- C) ReduceLROnPlateau
- D) TensorBoard

**9. What is the formula for Dice Coefficient?**
- A) Intersection / Union
- **B) (2 × Intersection) / (Prediction + Ground Truth)** ✓
- C) Prediction / Ground Truth
- D) Union / Intersection

**10. What optimizer is used in the notebook?**
- A) SGD
- B) RMSprop
- **C) Adam** ✓
- D) Adagrad

---

## **True/False Questions**

**11. U-Net was originally designed for medical image segmentation.**
- **True** ✓
- False

**12. The Dice Score and IoU always produce identical values.**
- True
- **False** ✓

**13. MaxPooling is used in the decoder path of U-Net.**
- True
- **False** ✓ (MaxPooling is used in the encoder path; Conv2DTranspose/upsampling is used in the decoder)

**14. BatchNormalization helps stabilize and speed up training.**
- **True** ✓
- False

**15. A higher IoU score indicates worse segmentation performance.**
- True
- **False** ✓ (Higher IoU = better performance)

---

## **Fill in the Blank**

**16. The loss function used combines Binary Cross-Entropy and _______ Loss.**
- **Answer: Dice**

**17. Images are normalized to the range [0, _____] before training.**
- **Answer: 1**

**18. The notebook uses a batch size of _______ for training.**
- **Answer: 16**

**19. The bottleneck layer has _______ filters (feature channels).**
- **Answer: 1024**

**20. Conv2DTranspose is used for _______ in the decoder path.**
- **Answer: upsampling** (or "upconvolution" / "transposed convolution")

---

## **Short Answer Questions**

**21. Explain the difference between IoU and Dice Score. When might one be preferred over the other?**

**Answer:** 
- IoU (Intersection over Union) = Intersection / Union
- Dice Score = (2 × Intersection) / (Prediction + Ground Truth)

Key differences:
- Dice Score gives more weight to true positives (2× in numerator)
- Dice is more sensitive to small objects
- IoU is stricter and penalizes false positives/negatives more heavily
- Dice Score might be preferred when dealing with imbalanced datasets or small segmentation targets
- IoU is commonly used as a standard metric for comparison across papers

**22. What is the purpose of the bottleneck layer in U-Net architecture?**

**Answer:** 
The bottleneck is the deepest layer in the U-Net that:
- Has the maximum number of feature channels (1024 in this implementation)
- Has the smallest spatial dimensions due to repeated downsampling
- Captures the highest-level abstract features and context
- Acts as a bridge between the encoder and decoder paths
- Contains the most compressed representation of the input image

**23. Why is a combined BCE + Dice loss used instead of just Binary Cross-Entropy?**

**Answer:**
The combined loss provides benefits from both:
- **BCE (Binary Cross-Entropy)**: Good for pixel-wise accuracy, treats each pixel independently
- **Dice Loss**: Considers spatial overlap, better for handling class imbalance, focuses on the region of interest

Together they:
- Balance pixel-level accuracy with overall segmentation quality
- Help with imbalanced datasets (where background >> foreground)
- Improve boundary detection and shape preservation
- Lead to better overall segmentation performance

**24. Describe what happens in an encoder block of the U-Net.**

**Answer:**
An encoder block consists of:
1. **Conv Block**: Two Conv2D layers (3×3 kernels) with BatchNormalization and ReLU activation
2. **MaxPooling2D**: (2×2) reduces spatial dimensions by half
3. **Returns two outputs**:
   - Features before pooling (saved for skip connection)
   - Pooled output (passed to next encoder block)

This progressively captures context while reducing spatial dimensions.

**25. What are the three main components of the U-Net architecture?**

**Answer:**
1. **Encoder (Contracting Path)**: Series of Conv→Conv→MaxPool blocks that downsample and extract features
2. **Bottleneck**: The deepest layer with maximum channels and minimum spatial dimensions
3. **Decoder (Expansive Path)**: Series of UpConv→Concatenate→Conv→Conv blocks that upsample and refine segmentation

Plus **Skip Connections** that link encoder and decoder to preserve spatial information.

---

## **Code Interpretation Questions**

**26. What does this line do: `mask = (mask > 0.5).astype(np.float32)`?**
- A) Normalizes the mask
- **B) Binarizes the mask** ✓
- C) Resizes the mask
- D) Inverts the mask

**Explanation:** Converts grayscale values to binary (0 or 1) by thresholding at 0.5

**27. In `ReduceLROnPlateau(factor=0.5, patience=5)`, what happens after 5 epochs without improvement?**
- A) Training stops
- **B) Learning rate is multiplied by 0.5** ✓
- C) Learning rate is divided by 5
- D) Batch size is reduced

**Explanation:** The learning rate is reduced by half (multiplied by 0.5) when validation loss doesn't improve for 5 consecutive epochs

**28. What does `smooth=1e-6` prevent in metric calculations?**
- A) Overfitting
- **B) Division by zero** ✓
- C) Gradient explosion
- D) Memory overflow

**Explanation:** Adds a small epsilon value to denominators to avoid division by zero errors when calculating IoU and Dice scores

---

## **Application Questions**

**29. If your model achieves 0.85 IoU on the test set, what does this indicate about its performance?**

**Answer:**
An IoU of 0.85 indicates **excellent performance**:
- 85% overlap between predicted and ground truth segmentations
- Generally, IoU > 0.7 is considered good
- IoU > 0.8 is considered very good/excellent
- IoU > 0.9 is exceptional
- For building segmentation (as in this dataset), 0.85 shows the model accurately identifies building boundaries with only 15% error

**30. Why would you use data augmentation in image segmentation tasks?**

**Answer:**
Data augmentation is used to:
- **Increase effective training data size** without collecting more images
- **Improve model generalization** to variations in real-world data
- **Prevent overfitting** by exposing the model to diverse examples
- **Handle limited datasets** (especially important in medical imaging)
- **Introduce invariance** to transformations like rotation, flipping, scaling, brightness changes

For segmentation, augmentation must be applied to **both image and mask simultaneously** to maintain correspondence.

---

## **Scoring Guide**

- **Multiple Choice (1-10)**: 10 questions × 2 points = 20 points
- **True/False (11-15)**: 5 questions × 2 points = 10 points
- **Fill in the Blank (16-20)**: 5 questions × 2 points = 10 points
- **Code Interpretation (26-28)**: 3 questions × 3 points = 9 points
- **Short Answer (21-25)**: 5 questions × 6 points = 30 points
- **Application (29-30)**: 2 questions × 10 points = 20 points

**Total: 99 points** (round to 100 for grading)

---

*Quiz based on: U-Net Image Segmentation with mIoU & Dice Score Tutorial*  
*Massachusetts Buildings Dataset - Kaggle*
