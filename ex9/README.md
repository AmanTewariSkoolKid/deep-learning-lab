# Object Detection Lab — R-CNN → Fast R-CNN (ex 9)

This folder contains a minimal Jupyter notebook and instructions to run a local lab that walks through an R-CNN → Fast R-CNN workflow using a local subset of the Kaggle Open-Images Bus & Truck dataset.

Files created by this step:
- `rcnn_minimal.ipynb` — a guided lab notebook scaffold (data checks, helpers, placeholders).\n

## Prerequisites

- Python 3.8+ installed.\n- (Optional but recommended) GPU with CUDA for faster training.\n
This guide uses PowerShell commands (Windows).

## Setup (PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
# activate
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies (if you have a `requirements.txt` at the repo root):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install minimal deps for the notebook:

```powershell
pip install jupyter pandas pillow matplotlib seaborn torchvision torch
```

Note: For PyTorch follow the official install instructions for CPU/GPU builds at https://pytorch.org.

3. (Optional tooling) If the repository provides `pre-commit` and lint config, you can install hooks:

```powershell
pip install pre-commit
pre-commit install
```


## Data placement

Place your local Open Images subset under `ex 9/data/` with the following expected structure:

```
ex 9/data/
  images/                  # image files (.jpg/.png)
  annotations/             # Open Images annotation CSV(s)
  class-descriptions.csv   # Open Images class mapping (MID -> display name)
```

Make sure `data/annotations/annotations.csv` exists and `data/class-descriptions.csv` exists. The notebook will attempt to resolve `Bus` and `Truck` MIDs from `class-descriptions.csv` and then filter the annotations.


## Quick runs

- Open the notebook (recommended) and run the cells in order:

```powershell
jupyter notebook "ex 9/rcnn_minimal.ipynb"
# or
jupyter lab
```

- Data filtering (inside the notebook): run the `filter_annotations(...)` helper cell to create `data/interim/annotations_oi_filtered.csv`.

- Generate selective search proposals (preferred: use the script provided by the project):

```powershell
python scripts/selective_search.py --images "ex 9/data/images" --out "ex 9/data/proposals"
```

If you don't have the script, the notebook prints placeholders and shows how to call a fallback.


## Training & evaluation

The prompt's Makefile suggests these targets. On Windows you can run the equivalent commands directly (PowerShell):

```powershell
# prepare data (if script present)
python scripts/prepare_openimages.py --base_dir "ex 9/data" --classes Bus Truck

# proposals
python scripts/selective_search.py --images "ex 9/data/images" --out "ex 9/data/proposals"

# train rcnn baseline (if trainer implemented)
python -m src.trainers.rcnn_trainer --config config/config.yml

# train fast rcnn head (if implemented)
python -m src.trainers.fast_rcnn_trainer --config config/config.yml

# evaluate
python -m src.eval.detect_eval --config config/config.yml
```

If your system does not have `make`, you can still run the commands above directly.


## Notes & next steps

- The notebook `rcnn_minimal.ipynb` is intentionally scaffolded and contains placeholders where the full implementations (selective search, proposal caching, training loops) are expected to live in `scripts/` and `src/` as per the project deliverables.\n
- Tell me if you'd like me to:\n  - Implement full `scripts/prepare_openimages.py` and the selective-search script, or\n  - Create a tiny synthetic dataset and an end-to-end smoke test that runs on CPU in <5 minutes.\n

## Contact/Help

Reply here with how you'd like to proceed and I will: implement the next piece, add tests, or wire up training commands so you can run an end-to-end demo on your machine.