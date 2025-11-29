# Object Detection using R-CNN Quiz - Answer Key

## **Multiple Choice Questions**

**1. What does R-CNN stand for?**
- A) Recurrent Convolutional Neural Network
- B) Regional Convolutional Neural Network
- **C) Region-based Convolutional Neural Network** ✓
- D) Recursive CNN

**2. What is the primary purpose of Selective Search in R-CNN?**
- A) To classify objects
- B) To train the neural network
- **C) To generate region proposals** ✓
- D) To compute loss functions

**3. What does IoU measure in object detection?**
- A) Image Quality
- B) Input Output Units
- **C) Intersection over Union** ✓
- D) Iterative Optimization Updates

**4. What is the IoU threshold used in the notebook to determine if a region proposal contains an object vs background?**
- A) 0.1
- B) 0.5
- **C) 0.3** ✓
- D) 0.7

**5. Which pre-trained model is used as the backbone in this R-CNN implementation?**
- A) ResNet
- B) AlexNet
- **C) VGG16** ✓
- D) MobileNet

**6. What loss function is used for classification in the R-CNN model?**
- A) Mean Squared Error
- B) Binary Cross Entropy
- **C) Cross Entropy Loss** ✓
- D) Hinge Loss

**7. What loss function is used for bounding box regression?**
- A) Cross Entropy Loss
- **B) L1 Loss** ✓
- C) Dice Loss
- D) Focal Loss

**8. What does NMS stand for?**
- A) Network Model Selection
- B) Normalized Mean Squared
- **C) Non-Maximum Suppression** ✓
- D) Neural Model Sampling

**9. What is the purpose of NMS in object detection?**
- A) To improve accuracy
- B) To speed up training
- **C) To eliminate duplicate/overlapping bounding boxes** ✓
- D) To normalize the input

**10. What is the input image size that crops are resized to before feeding into the VGG backbone?**
- A) 128x128
- B) 512x512
- **C) 224x224** ✓
- D) 256x256

**11. What activation function is used in the bounding box regression head?**
- A) Sigmoid
- B) ReLU
- **C) Tanh** ✓
- D) Softmax

**12. Which optimizer is used for training the R-CNN model?**
- A) Adam
- **B) SGD** ✓
- C) RMSprop
- D) Adagrad

**13. What is the lambda (weight) value for regression loss in the combined loss?**
- A) 1.0
- B) 5.0
- **C) 10.0** ✓
- D) 0.1

**14. How are the bounding box coordinates represented in the dataset?**
- A) x, y, width, height
- **B) XMin, YMin, XMax, YMax** ✓
- C) center_x, center_y, width, height
- D) Only corners

**15. What percentage of data is used for training in the train-test split?**
- A) 80%
- **B) 90%** ✓
- C) 70%
- D) 85%

---

## **True/False Questions**

**16. R-CNN processes the entire image at once to detect objects.**
- True
- **False** ✓ (R-CNN processes region proposals separately)

**17. Selective Search groups pixels based on color, texture, size, and shape compatibility.**
- **True** ✓
- False

**18. In the notebook, regions smaller than 5% of the image area are discarded.**
- **True** ✓
- False

**19. The VGG backbone parameters are trainable during R-CNN training.**
- True
- **False** ✓ (Parameters are frozen with `requires_grad = False`)

**20. Bounding box offsets (deltas) are normalized by the image dimensions.**
- **True** ✓
- False

**21. The background class is included in bounding box regression loss calculation.**
- True
- **False** ✓ (Background class is excluded from regression loss)

**22. Non-Maximum Suppression uses an IoU threshold of 0.05 in this implementation.**
- **True** ✓
- False

**23. R-CNN can detect multiple objects in a single image.**
- **True** ✓
- False

**24. The model outputs both class probabilities and bounding box offsets.**
- **True** ✓
- False

**25. Softmax is applied to the classification output during inference.**
- **True** ✓
- False

---

## **Fill in the Blank**

**26. Selective Search parameters include scale=_____ and min_size=_____.**
- **Answer: 200, 100**

**27. The feature dimension from VGG16 backbone is _____.**
- **Answer: 25088**

**28. The bounding box regression head has _____ output neurons.**
- **Answer: 4** (for x, y, width, height offsets)

**29. Region proposals with IoU > 0.3 are labeled as _____, otherwise as _____.**
- **Answer: object class, background**

**30. The model is trained for _____ epochs in the notebook.**
- **Answer: 5**

**31. Images are normalized using mean=[0.485, 0.456, 0.406] and std=[_____, _____, _____].**
- **Answer: 0.229, 0.224, 0.225**

**32. The bounding box regression head contains a hidden layer with _____ neurons.**
- **Answer: 512**

**33. During preprocessing, images are converted from BGR to _____ color space.**
- **Answer: RGB**

**34. The NMS IoU threshold used to eliminate duplicate boxes is _____.**
- **Answer: 0.05**

**35. The batch size used for training is _____.**
- **Answer: 2**

---

## **Short Answer Questions**

**36. Explain the six main steps involved in training an R-CNN object detection model.**

**Answer:**
1. **Creating ground truth data**: Labels containing bounding boxes and class for objects in images
2. **Region proposals**: Use Selective Search to identify regions likely to contain objects
3. **Creating target class variable**: Use IoU metric to determine if proposal contains an object or background
4. **Creating target bounding box offsets**: Calculate delta differences between ground truth and region proposals
5. **Building the model**: Create a model that predicts both object class and bounding box offsets
6. **Measuring accuracy**: Evaluate using mean Average Precision (mAP) metric

**37. How does Selective Search generate region proposals? What factors does it consider?**

**Answer:**
Selective Search generates region proposals through hierarchical grouping of similar pixels. It considers:
- **Color**: Pixel intensity similarities
- **Texture**: Patterns in the image
- **Size**: Region dimensions
- **Shape compatibility**: How well regions fit together

The algorithm uses a bottom-up approach, starting with individual pixels and progressively merging similar regions based on these features. Parameters like `scale=200` and `min_size=100` control the granularity of proposals.

**38. What is the formula for calculating IoU? Why is it important in object detection?**

**Answer:**
**Formula**: IoU = (Area of Intersection) / (Area of Union)

Where:
- Intersection = Overlap between predicted box and ground truth box
- Union = Combined area of both boxes minus the intersection

**Importance**:
- Measures how well predicted boxes match ground truth
- Used to determine if region proposal should be labeled as object (IoU > 0.3) or background
- Used in NMS to eliminate duplicate detections
- Standard metric for evaluating object detection performance
- Range: [0, 1], where 1 = perfect overlap

**39. Describe the architecture of the R-CNN model. What are its main components?**

**Answer:**
The R-CNN architecture consists of:

1. **VGG16 Backbone**: 
   - Pre-trained feature extractor
   - Parameters frozen (not trained)
   - Outputs 25088-dimensional features
   - Classifier layers removed

2. **Classification Head**:
   - Single linear layer
   - Outputs class scores for all classes (including background)
   - Uses Cross Entropy Loss

3. **Bounding Box Regression Head**:
   - Linear(25088 → 512) + ReLU
   - Linear(512 → 4) + Tanh
   - Outputs 4 offsets (Δx, Δy, ΔX, ΔY)
   - Uses L1 Loss

**40. Explain the purpose of bounding box delta/offset regression. How are deltas calculated and used?**

**Answer:**
**Purpose**: Region proposals from Selective Search are approximate. Delta regression refines these proposals to better match ground truth bounding boxes.

**Calculation**:
```
delta = [gt_x - prop_x, gt_y - prop_y, gt_X - prop_X, gt_Y - prop_Y]
normalized_delta = delta / [W, H, W, H]
```

**Usage**:
1. During training: Model learns to predict deltas between proposals and ground truth
2. During inference: Predicted deltas are added to region proposals
   ```
   final_bbox = region_proposal + predicted_delta
   ```
3. Normalized by image dimensions to make learning easier
4. Tanh activation ensures deltas stay in reasonable range [-1, 1]

**41. What is Non-Maximum Suppression and why is it necessary? Describe how it works.**

**Answer:**
**What**: NMS is a post-processing technique to eliminate duplicate/overlapping bounding boxes for the same object.

**Why Necessary**: 
- Multiple region proposals often overlap the same object
- Without NMS, same object detected multiple times
- Results in cleaner, more accurate detections

**How it Works**:
1. Sort all bounding boxes by confidence score (descending)
2. Select box with highest confidence
3. Calculate IoU between this box and all remaining boxes
4. Remove boxes with IoU > threshold (0.05 in this implementation)
5. Repeat with next highest confidence box
6. Continue until all boxes processed

Result: Only the best box per object remains.

**42. Why are the VGG16 backbone parameters frozen during training?**

**Answer:**
Reasons for freezing VGG16 parameters:

1. **Transfer Learning**: VGG16 is pre-trained on ImageNet, already learned powerful feature representations
2. **Computational Efficiency**: Fewer parameters to update = faster training, less memory
3. **Preventing Overfitting**: With limited training data, updating all parameters could cause overfitting
4. **Feature Preservation**: Pre-trained features are general and useful for many tasks
5. **Focus on Task-Specific Layers**: Only classification and regression heads need to learn domain-specific patterns

Code: `param.requires_grad = False` and `vgg_backbone.eval()` ensure parameters don't update.

**43. Explain the collate_fn function. What preprocessing steps does it perform?**

**Answer:**
The `collate_fn` processes a batch of images and prepares data for the model:

**Steps**:
1. **Extract crops**: Gets all region proposal crops from all images in batch
2. **Resize**: Resizes each crop to 224×224 (VGG input size)
3. **Normalize**: 
   - Divides by 255 to get [0, 1] range
   - Applies ImageNet normalization (mean/std)
   - Converts to tensor
4. **Concatenate**: Combines all crops into single batch tensor
5. **Process labels**: Converts class names to integer targets using `label2target` dictionary
6. **Process deltas**: Converts bounding box offsets to tensors
7. **Move to device**: Transfers everything to GPU/CPU

Result: Model-ready batches of crops, labels, and deltas.

**44. What is the combined loss function in R-CNN? Why use a weighted combination?**

**Answer:**
**Combined Loss**:
```
Total Loss = Detection Loss + λ × Regression Loss
```
Where λ = 10.0

**Components**:
- **Detection Loss**: Cross Entropy Loss for classification
- **Regression Loss**: L1 Loss for bounding box offsets

**Why Weighted Combination**:
1. **Multi-task Learning**: Model must perform two tasks simultaneously (classification + localization)
2. **Scale Balancing**: Classification and regression losses have different scales
3. **Priority Control**: λ = 10.0 gives more importance to accurate localization
4. **Better Performance**: Joint optimization improves both tasks
5. **Exclusion of Background**: Regression loss only calculated for non-background classes (sensible since background has no bbox)

**Calculation**: Background detections excluded from regression loss using `ixs = torch.where(labels != 1)`.

**45. How does the model handle the class imbalance between background and object classes?**

**Answer:**
Class imbalance handling strategies in the notebook:

1. **Selective IoU Threshold**: 
   - Only proposals with IoU > 0.3 labeled as objects
   - Most proposals become background
   - Creates more conservative positive samples

2. **Background Exclusion from Regression**:
   - Background class excluded from bbox regression loss
   - `ixs = torch.where(labels != 1)` filters out background
   - Focuses regression learning on actual objects

3. **Implicit Handling**:
   - Cross Entropy Loss naturally handles imbalanced classes
   - Model learns background is common
   - Confidence thresholding during inference helps

**Note**: More advanced approaches could include:
- Focal Loss
- Hard negative mining
- Balanced sampling
- Class weights

---

## **Code Interpretation Questions**

**46. What does this line do: `candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])`?**
- A) Normalizes the candidates
- **B) Converts (x, y, width, height) format to (x1, y1, x2, y2) format** ✓
- C) Filters out small candidates
- D) Resizes the candidates

**Explanation**: Transforms bounding box representation from top-left corner + dimensions to top-left + bottom-right corners.

**47. What does `img[...,::-1]` accomplish?**
- A) Flips image horizontally
- B) Flips image vertically
- **C) Converts BGR to RGB color channels** ✓
- D) Normalizes the image

**Explanation**: OpenCV reads images as BGR; `[::-1]` reverses the last dimension to get RGB.

**48. In `crops = [cv2.resize(crop, (224,224)) for crop in crops]`, why 224×224?**
- A) It's a random choice
- B) It reduces computation time
- **C) VGG16 expects 224×224 input** ✓
- D) It's the original image size

**Explanation**: VGG16 architecture was trained on 224×224 ImageNet images.

**49. What does `torch.where(labels != 1)` return?**
- A) Boolean mask
- **B) Indices where condition is true** ✓
- C) Count of matching elements
- D) Modified labels tensor

**Explanation**: Returns indices of all non-background samples (assuming background has label 1).

**50. What happens in `_, _clss = _clss.max(-1)`?**
- A) Finds minimum class
- B) Normalizes classes
- **C) Gets class index with maximum probability** ✓
- D) Counts number of classes

**Explanation**: `max(-1)` returns (max_values, max_indices) along last dimension; `_` discards values, keeps indices.

**51. What does `img.permute(2,0,1)` do?**
- A) Rotates the image
- B) Flips color channels
- **C) Changes from HWC to CHW format** ✓
- D) Normalizes dimensions

**Explanation**: PyTorch expects [Channels, Height, Width] format; permute rearranges from [H, W, C].

**52. Why is `drop_last=True` used in DataLoader?**
- A) To remove corrupted data
- B) To speed up training
- **C) To ensure consistent batch sizes** ✓
- D) To reduce memory usage

**Explanation**: Drops incomplete last batch if dataset size isn't divisible by batch size.

---

## **Application Questions**

**53. If a region proposal has IoU of 0.25 with ground truth, how would it be labeled and why?**

**Answer:**
It would be labeled as **"background"**.

**Reasoning**:
- The notebook uses IoU threshold of 0.3
- 0.25 < 0.3, so it doesn't meet the object threshold
- Low IoU indicates poor overlap with actual object
- This conservative threshold helps reduce false positives
- Such proposals are used as negative training examples

**54. Why does the model use both classification and regression heads instead of just classification?**

**Answer:**
Both are necessary for complete object detection:

**Classification Head**:
- Identifies WHAT the object is
- Determines if region contains object or background
- Outputs class probabilities

**Regression Head**:
- Identifies WHERE the object is precisely
- Refines bounding box location
- Corrects imperfect region proposals from Selective Search

**Example**: Selective Search might propose a box slightly off-center. Classification says "this is a bus," but regression adjusts the box to perfectly fit the bus.

Without regression: Bounding boxes would be inaccurate.
Without classification: We wouldn't know what's in the boxes.

**55. Explain how you would use this trained R-CNN model to detect objects in a new image.**

**Answer:**
**Inference Pipeline**:

1. **Load and preprocess image**: Read image, convert BGR→RGB
2. **Generate region proposals**: Run Selective Search on the image
3. **Extract crops**: For each proposal, extract and resize to 224×224
4. **Preprocess crops**: Normalize using ImageNet mean/std
5. **Forward pass**: Feed crops through model
   - Get class probabilities and delta predictions
6. **Apply softmax**: Convert logits to probabilities
7. **Filter background**: Remove proposals classified as background
8. **Apply deltas**: Add predicted offsets to region proposals
9. **Apply NMS**: Eliminate duplicate detections (IoU threshold 0.05)
10. **Select best detection**: Choose box with highest confidence
11. **Visualize**: Draw bounding box with class label on image

**Code**: Use the `test_predictions(filename)` function from the notebook.

**56. What are the limitations of R-CNN compared to modern object detectors?**

**Answer:**
**R-CNN Limitations**:

1. **Speed**: 
   - Very slow (processes each region proposal separately)
   - Selective Search is slow
   - Thousands of forward passes per image
   - Not suitable for real-time applications

2. **Multi-stage Training**: 
   - Complex training pipeline
   - Selective Search not learned end-to-end
   - Separate classification and regression training

3. **Fixed Region Proposals**: 
   - Selective Search quality limits detection quality
   - Can't adapt proposals during training

4. **Storage**: 
   - Features must be cached during training
   - High disk space requirements

5. **No sharing of computation**: Each proposal processed independently

**Modern Solutions**:
- **Fast R-CNN**: Shares computation, single-stage training
- **Faster R-CNN**: Learned region proposals (RPN)
- **YOLO/SSD**: Single-shot detectors, much faster
- **EfficientDet**: Better speed-accuracy tradeoff

**57. How would you improve the model's performance if it was achieving low accuracy?**

**Answer:**
**Improvement Strategies**:

1. **Data Augmentation**:
   - Horizontal flipping
   - Random crops
   - Color jittering
   - Rotation (with bbox transformation)

2. **Hyperparameter Tuning**:
   - Increase epochs (5 is quite low)
   - Adjust learning rate
   - Modify lambda weight
   - Change batch size
   - Tune IoU thresholds

3. **Architecture Changes**:
   - Use ResNet instead of VGG16 (better features)
   - Deeper regression head
   - Batch normalization in heads
   - Different activation functions

4. **Training Improvements**:
   - Use Adam optimizer instead of SGD
   - Learning rate scheduling
   - Gradient clipping
   - Fine-tune backbone (unfreeze some layers)

5. **Data Quality**:
   - More training data
   - Better quality annotations
   - Class balancing
   - Hard negative mining

6. **Loss Function**:
   - Smooth L1 Loss instead of L1
   - Focal Loss for classification
   - Different lambda values

7. **Post-processing**:
   - Adjust NMS threshold
   - Confidence thresholding
   - Multi-scale testing

**58. Why is the regression loss multiplied by 10.0 (lambda) but detection loss is not?**

**Answer:**
**Reasoning for λ = 10.0**:

1. **Scale Mismatch**: 
   - Classification loss (Cross Entropy) typically ranges 0-3
   - Regression loss (L1) for normalized deltas is typically 0-0.3
   - Without weighting, regression contributes too little

2. **Importance Balancing**:
   - Precise localization is crucial for object detection
   - λ = 10 makes regression equally important as classification
   - Ensures model doesn't just focus on classification

3. **Gradient Magnitude**:
   - Classification gradients naturally larger
   - Scaling regression loss balances gradient magnitudes
   - Helps both tasks learn effectively

4. **Empirical Finding**:
   - λ = 10 is a common value in literature
   - Works well across different datasets

**Without Lambda**: Model would prioritize classification, producing accurate classes but poor bounding boxes.

---

## **Debugging/Troubleshooting Questions**

**59. During training, you notice the regression loss stays high but classification loss decreases. What might be wrong?**

**Answer:**
**Possible Issues**:

1. **Lambda too small**: Regression not weighted enough in total loss
2. **Poor region proposals**: Selective Search generating bad proposals with low IoU
3. **Learning rate**: Too low for regression head
4. **Tanh saturation**: Deltas hitting [-1, 1] limits
5. **Normalization issues**: Deltas not properly normalized by image dimensions
6. **Insufficient training**: Need more epochs for regression to converge
7. **Architecture**: Regression head too shallow (needs more capacity)

**Solutions**:
- Increase lambda weight
- Check delta value distributions
- Use separate learning rates
- Add more layers to regression head
- Verify bbox preprocessing

**60. What would happen if you forgot to exclude background class from regression loss?**

**Answer:**
**Problems**:

1. **Meaningless Learning**: 
   - Background has no bounding box
   - Model tries to predict deltas for non-existent objects
   - Learns garbage for background samples

2. **Diluted Gradients**:
   - Most proposals are background
   - Real object regression gradients overwhelmed
   - Slower convergence for actual objects

3. **Poor Localization**:
   - Model learns average of meaningful and meaningless deltas
   - Bounding boxes become less accurate

4. **Increased Loss**:
   - Background delta "errors" artificially inflate loss
   - Harder to assess true performance

**Fix**: `ixs = torch.where(labels != background_class)` filters properly.

---

## **Conceptual Deep-Dive Questions**

**61. Compare and contrast R-CNN with Fast R-CNN and Faster R-CNN.**

**Answer:**

| Feature | R-CNN | Fast R-CNN | Faster R-CNN |
|---------|-------|------------|--------------|
| **Region Proposals** | Selective Search (external) | Selective Search (external) | RPN (learned) |
| **Feature Extraction** | Per-proposal CNN | Shared CNN + RoI pooling | Shared CNN + RoI pooling |
| **Training** | Multi-stage | Single-stage | Single-stage (end-to-end) |
| **Speed** | Very slow (~47s/image) | Fast (~2s/image) | Faster (~0.2s/image) |
| **Storage** | Requires feature caching | No caching needed | No caching needed |
| **Accuracy** | Good | Better | Best |

**Key Improvements**:
- **Fast R-CNN**: Shares computation via RoI pooling
- **Faster R-CNN**: Learns proposals via Region Proposal Network (RPN)

**62. Explain the concept of transfer learning as applied in this R-CNN implementation.**

**Answer:**
**Transfer Learning in R-CNN**:

**What**: Using pre-trained VGG16 weights instead of training from scratch.

**How it Works**:
1. **Source Task**: VGG16 trained on ImageNet (1000 classes, 1.2M images)
2. **Transfer**: Use learned weights as feature extractor
3. **Target Task**: Object detection on custom dataset
4. **Fine-tuning**: Train only classification and regression heads

**Benefits**:
- **Better Features**: Low-level features (edges, textures) transfer well
- **Less Data Needed**: Don't need millions of images
- **Faster Training**: Only train small heads
- **Better Performance**: Pre-trained features generalize well

**Why It Works**:
- Early CNN layers learn universal features
- Object detection and classification share similar low-level patterns
- ImageNet diversity provides robust representations

**Code**: `pretrained=True` loads ImageNet weights; `requires_grad=False` freezes them.

---

## **Scoring Guide**

### **Point Distribution**
- **Multiple Choice (1-15)**: 15 questions × 2 points = **30 points**
- **True/False (16-25)**: 10 questions × 1 point = **10 points**
- **Fill in the Blank (26-35)**: 10 questions × 2 points = **20 points**
- **Code Interpretation (46-52)**: 7 questions × 3 points = **21 points**
- **Short Answer (36-45)**: 10 questions × 5 points = **50 points**
- **Application (53-58)**: 6 questions × 5 points = **30 points**
- **Debugging (59-60)**: 2 questions × 5 points = **10 points**
- **Conceptual Deep-Dive (61-62)**: 2 questions × 5 points = **10 points**

**Total: 181 points**

### **Grading Scale (Normalized to 100%)**
- **A (90-100%)**: 163-181 points
- **B (80-89%)**: 145-162 points
- **C (70-79%)**: 127-144 points
- **D (60-69%)**: 109-126 points
- **F (<60%)**: <109 points

---

*Quiz based on: Object Detection using R-CNN Tutorial*  
*Dataset: OpenImages subset for object detection*  
*Architecture: VGG16 backbone + Classification + Regression heads*
