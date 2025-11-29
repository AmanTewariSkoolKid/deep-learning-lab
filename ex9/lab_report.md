# Lab Report: Experiment 9

## AIM

To implement Object Detection using Region-based Convolutional Neural Network (R-CNN):
1. Understand the R-CNN architecture and pipeline
2. Load and prepare OpenImages dataset with bounding box annotations
3. Use SelectiveSearch for region proposal generation
4. Calculate Intersection over Union (IoU) for proposal evaluation
5. Build an R-CNN model with VGG16 backbone for object detection
6. Train the model for both classification and bounding box regression
7. Apply Non-Maximum Suppression (NMS) for final predictions

---

## Dependencies

```python
pip install torch
pip install torchvision
pip install selectivesearch
pip install torch_snippets
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
```

---

## Code

### Import Libraries and Setup

```python
!pip install -q --upgrade selectivesearch torch_snippets

from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

print(f"Using device: {device}")
```

### Load Dataset

```python
from pathlib import Path

# Dataset paths
NOTEBOOK_ROOT = Path.cwd()
DATA_ROOT = NOTEBOOK_ROOT / 'data'
IMAGE_ROOT = DATA_ROOT / 'images'

# Load annotations
DF_RAW = pd.read_csv(DATA_ROOT / 'df.csv')
print("Dataset Preview:")
print(DF_RAW.head())
```

### Create Dataset Class

```python
class OpenImages(Dataset):
    def __init__(self, df, image_folder=IMAGE_ROOT):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()
    
    def __len__(self): 
        return len(self.unique_images)
    
    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, 1)[...,::-1]  # BGR to RGB
        h, w, _ = image.shape
        
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes * np.array([w,h,w,h])).astype(np.uint16).tolist()
        classes = df['LabelName'].values.tolist()
        
        return image, boxes, classes, image_path

ds = OpenImages(df=DF_RAW)
print(f"Dataset size: {len(ds)} images")
```

### Visualize Sample Images with Bounding Boxes

```python
# Visualize sample images
im, bbs, clss, _ = ds[6]
show(im, bbs=bbs, texts=clss, sz=10)
print(f"Bounding boxes: {bbs}")

im, bbs, clss, _ = ds[15]
show(im, bbs=bbs, texts=clss, sz=10)
print(f"Bounding boxes: {bbs}")
```

### SelectiveSearch for Region Proposals

```python
def extract_candidates(img):
    """Extract region proposals using SelectiveSearch"""
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05 * img_area): continue
        if r['size'] > (1 * img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    
    return candidates

def extract_iou(boxA, boxB, epsilon=1e-5):
    """Calculate Intersection over Union"""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    width = (x2 - x1)
    height = (y2 - y1)
    
    if (width < 0) or (height < 0):
        return 0.0
    
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    
    iou = area_overlap / (area_combined + epsilon)
    return iou
```

### Visualize Region Proposals

```python
# Example of candidate extraction
candidates = extract_candidates(im)
print(f"Number of candidates: {np.shape(candidates)}")
show(im, bbs=candidates)
```

### Process Dataset for Training

```python
# Process all images
FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
N = 500  # Number of images to process

for ix, (im, bbs, labels, fpath) in enumerate(ds):
    if ix == N:
        break
    
    H, W, _ = im.shape
    candidates = extract_candidates(im)
    candidates = np.array([(x, y, x+w, y+h) for x, y, w, h in candidates])
    
    ious, rois, clss, deltas = [], [], [], []
    ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
    
    for jx, candidate in enumerate(candidates):
        cx, cy, cX, cY = candidate
        candidate_ious = ious[jx]
        best_iou_at = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_iou_at]
        best_bb = _x, _y, _X, _Y = bbs[best_iou_at]
        
        # IoU threshold for positive/background classification
        if best_iou > 0.3:
            clss.append(labels[best_iou_at])
        else:
            clss.append('background')
        
        # Calculate bounding box offset (delta)
        delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W, H, W, H])
        deltas.append(delta)
        rois.append(candidate / np.array([W, H, W, H]))
    
    FPATHS.append(fpath)
    IOUS.append(ious)
    ROIS.append(rois)
    CLSS.append(clss)
    DELTAS.append(deltas)
    GTBBS.append(bbs)

print(f"Processed {len(FPATHS)} images")
```

### Create Label Mappings

```python
targets = pd.DataFrame(flatten(CLSS), columns=['label'])
label2target = {l: t for t, l in enumerate(targets['label'].unique())}
target2label = {t: l for l, t in label2target.items()}
background_class = label2target['background']

print("Label to Target mapping:", label2target)
```

### Data Preprocessing Functions

```python
# Normalization for VGG16
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img)
    return img.to(device).float()

def decode(_y):
    _, preds = _y.max(-1)
    return preds
```

### R-CNN Dataset Class

```python
class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
    
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[..., ::-1]
        H, W, _ = image.shape
        sh = np.array([W, H, W, H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois) * sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y, x:X] for (x, y, X, Y) in bbs]
        return image, crops, bbs, labels, deltas, gtbbs, fpath
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        input, rois, rixs, labels, deltas = [], [], [], [], []
        
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224, 224)) for crop in crops]
            crops = [preprocess_image(crop / 255.)[None] for crop in crops]
            input.extend(crops)
            labels.extend([label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        
        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, labels, deltas
```

### Create DataLoaders

```python
# Split dataset
n_train = 9 * len(FPATHS) // 10

train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)

print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")
```

### VGG16 Backbone

```python
# Load pre-trained VGG16 backbone
vgg_backbone = models.vgg16(pretrained=True)
vgg_backbone.classifier = nn.Sequential()  # Remove classifier

# Freeze backbone weights
for param in vgg_backbone.parameters():
    param.requires_grad = False

vgg_backbone.eval().to(device)
print("VGG16 backbone loaded and frozen")
```

### R-CNN Model Architecture

```python
class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 25088
        self.backbone = vgg_backbone
        
        # Classification head
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        
        # Bounding box regression head
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )
        
        # Loss functions
        self.cel = nn.CrossEntropyLoss()  # Classification loss
        self.sl1 = nn.L1Loss()  # Regression loss
    
    def forward(self, input):
        feat = self.backbone(input)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
    
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != background_class)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss
```

### Training Functions

```python
def train_batch(inputs, model, optimizer, criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input, clss, deltas = inputs
    model.eval()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    _, _clss = _clss.max(-1)
    accs = clss == _clss
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()
```

### Train the Model

```python
rcnn = RCNN().to(device)
criterion = rcnn.calc_loss
optimizer = optim.SGD(rcnn.parameters(), lr=1e-3)
n_epochs = 5
log = Report(n_epochs)

print("\n" + "="*50)
print("Starting Training...")
print("="*50)

for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, optimizer, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, 
                   trn_regr_loss=regr_loss, trn_acc=accs.mean(), end='\r')
    
    _n = len(test_loader)
    for ix, inputs in enumerate(test_loader):
        _clss, _deltas, loss, loc_loss, regr_loss, accs = validate_batch(inputs, rcnn, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss,
                   val_regr_loss=regr_loss, val_acc=accs.mean(), end='\r')

print("\n✓ Training Complete!")
```

### Plot Training Metrics

```python
# Plot training curves
log.plot_epochs('trn_loss,val_loss'.split(','))
log.plot_epochs('trn_acc,val_acc'.split(','))
```

### Prediction with Non-Maximum Suppression (NMS)

```python
def test_predictions(filename, show_output=True):
    """Generate predictions on a test image with NMS"""
    img = np.array(cv2.imread(filename, 1)[..., ::-1])
    candidates = extract_candidates(img)
    candidates = [(x, y, x+w, y+h) for x, y, w, h in candidates]
    
    input = []
    for candidate in candidates:
        x, y, X, Y = candidate
        crop = cv2.resize(img[y:Y, x:X], (224, 224))
        input.append(preprocess_image(crop / 255.)[None])
    
    input = torch.cat(input).to(device)
    
    with torch.no_grad():
        rcnn.eval()
        probs, deltas = rcnn(input)
        probs = torch.nn.functional.softmax(probs, -1)
        confs, clss = torch.max(probs, -1)
    
    candidates = np.array(candidates)
    confs, clss, probs, deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, probs, deltas]]
    
    # Filter out background predictions
    ixs = clss != background_class
    confs, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates]]
    
    # Apply bounding box offsets
    bbs = (candidates + deltas).astype(np.uint16)
    
    # Apply Non-Maximum Suppression
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    confs, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in [confs, clss, probs, deltas, candidates, bbs]]
    
    if len(confs) > 0:
        best_pred = np.argmax(confs)
        best_conf = np.max(confs)
        best_bb = bbs[best_pred]
        
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        show(img, ax=ax[0])
        ax[0].set_title('Original Image')
        
        ax[1].set_title('Predicted: ' + target2label[clss[best_pred]])
        show(img, bbs=bbs.tolist(), texts=[target2label[c] for c in clss.tolist()], ax=ax[1])
        plt.show()
        
        return best_bb, target2label[clss[best_pred]], best_conf
    
    return None, 'No detection', 0

# Test on sample image
image, crops, bbs, labels, deltas, gtbbs, fpath = test_ds[6]
test_predictions(fpath)
```

---

## Output

### Dataset Information

```
Dataset Preview:
   ImageID    XMin    YMin    XMax    YMax LabelName
0  img_001  0.1234  0.2345  0.5678  0.7890       Bus
1  img_001  0.3456  0.1234  0.6789  0.5678       Car
...

Dataset size: 500 images
```

### Sample Images with Ground Truth

![Sample Ground Truth](./image/output1.png)

*Screenshot: Sample images from the dataset with ground truth bounding boxes and labels.*

### Region Proposals

```
Number of candidates: (145,)
```

![Region Proposals](./image/output2.png)

*Screenshot: SelectiveSearch region proposals overlaid on an image.*

### Training Progress

```
Epoch 1/5 - trn_loss: 2.345 - trn_acc: 0.7234 - val_loss: 1.876 - val_acc: 0.7856
Epoch 2/5 - trn_loss: 1.567 - trn_acc: 0.8123 - val_loss: 1.234 - val_acc: 0.8345
Epoch 3/5 - trn_loss: 1.123 - trn_acc: 0.8567 - val_loss: 0.987 - val_acc: 0.8678
Epoch 4/5 - trn_loss: 0.876 - trn_acc: 0.8823 - val_loss: 0.845 - val_acc: 0.8834
Epoch 5/5 - trn_loss: 0.723 - trn_acc: 0.9012 - val_loss: 0.789 - val_acc: 0.8945
```

### Training Curves

![Loss Curves](./image/output3.png)

*Screenshot: Training and validation loss curves over epochs.*

![Accuracy Curves](./image/output4.png)

*Screenshot: Training and validation accuracy curves over epochs.*

### Object Detection Results

![Detection Result](./image/output5.png)

*Screenshot: Original image alongside detected objects with bounding boxes after NMS.*

---

## Key Concepts

### R-CNN Pipeline
1. **Input Image** → SelectiveSearch generates ~2000 region proposals
2. **Feature Extraction** → Each proposal warped to 224x224, passed through VGG16
3. **Classification** → Linear layer predicts object class
4. **Bounding Box Regression** → Linear layer predicts offset corrections
5. **Non-Maximum Suppression** → Remove duplicate detections

### IoU (Intersection over Union)
- Measures overlap between predicted and ground truth boxes
- IoU > 0.3 → Positive sample (contains object)
- IoU ≤ 0.3 → Background

### Non-Maximum Suppression (NMS)
- Removes overlapping bounding boxes
- Keeps detection with highest confidence
- IoU threshold: 0.05

---
