# Notes — R-CNN minimal lab (layman-friendly)

This `notes.md` summarizes the essential ideas and the quick steps to run the recreated notebook `rcnn_recreated.ipynb`.

What this lab does (high level)
- Dataset: a small local subset of Open Images with two classes (Bus and Truck). The annotations are expected under `ex 9/data/interim/annotations_oi_filtered_pixels.csv` with pixel coordinates.
- Proposal stage: selective search generates many candidate boxes per image that might contain objects.
- R-CNN pipeline: crop each proposal, resize to a fixed size, run a pretrained CNN (VGG) to get features, and train small heads:
  - A classifier head to predict the class for each proposal (Bus, Truck, or background).
  - A regression head to predict box offsets (to refine the proposal into a better box).

Key concepts (plain language)
- Region proposals: instead of sliding a small window everywhere, we use an algorithm (selective search) that proposes a few hundred regions per image that are likely to be objects. This reduces the amount of work.
- IoU (Intersection over Union): measures how much two boxes overlap (0 means none, 1 means exact). We use IoU to decide if a proposal matches a ground-truth box (e.g. IoU >= 0.5 is a positive match).
- Background: proposals with low IoU to any GT box are labeled as background and used to train the classifier to ignore them.
- Delta encoding: the regression head learns a simple offset between the proposal and the ground-truth box. After training, we add the predicted offset to the proposal to get a refined bounding box.

Files the notebook expects / creates
- Input: `ex 9/data/images/` (images), `ex 9/data/interim/annotations_oi_filtered_pixels.csv` (pixel-coordinates annotation per image)
- Outputs (demo): in-memory `small_ds` (list of proposals), optional `ex 9/data/splits/*.txt` if you run the other notebook's split functions.

Quick run steps (PowerShell)
1. Activate your Python environment (example):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install packages (only if missing):

```powershell
pip install selectivesearch torch torchvision pandas pillow matplotlib scikit-learn opencv-python
```

3. Start Jupyter and open the recreated notebook:

```powershell
jupyter notebook "ex 9/rcnn_recreated.ipynb"
```

4. Run cells from top to bottom. The notebook is intentionally conservative — it builds a small proposal dataset and runs a single training batch as a smoke test. If all runs without error, you have a working minimal R-CNN pipeline.

Troubleshooting tips
- If the pixel CSV is missing: run `ex 9/rcnn_minimal.ipynb` first and run the `filter_annotations_to_mids(...)` and `convert_normalized_to_pixels(...)` functions to create `ex 9/data/interim/annotations_oi_filtered_pixels.csv`.
- If images are not found: verify `ex 9/data/images/` contains files named by `ImageID` (Kaggle export typical), or inspect the filenames and tell me the naming pattern so I can adapt the lookup.
- Slow selective search: reduce `N_images` or reduce `max_props_per_image` in the building step.

Next steps I can implement for you
- `scripts/prepare_openimages.py`: a CLI to perform MID resolution, normalized->pixel conversion and produce splits.
- `scripts/selective_search.py`: a script to generate proposals and cache them as JSON/NPY per image.
- Full trainer and evaluation: multi-epoch training, model checkpointing, mAP evaluation and PR curves.

Tell me which next step you want and I'll implement it and update the todo list.