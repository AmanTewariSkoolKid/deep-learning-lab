# Deep Learning Lab

A structured collection of laboratory exercises (ex1–ex10) covering core deep learning workflows: data preparation, CNNs, transfer learning, segmentation (UNet), object detection (RCNN), and experimentation with various datasets. This repository consolidates code, notebooks, reports, and trained model artifacts generated during a Semester 5 Deep Learning Lab.

## Repository Structure
```
ex1/  Intro & initial experimentation
ex2/  Data handling & basic model training
ex3/  Exercise 3 experiments (notebook + report)
ex4/  Exercise 4 experiments (notebook + report)
ex5/  Transfer learning on flower dataset
ex6/  Exercise 6 experiments
ex7/  CNN tutorial & MNIST model
ex8/  Makeup vs No-Makeup classification dataset
ex9/  RCNN object detection + analytics data
ex10/ UNet segmentation (medical/annotation style)
```
Each folder contains: notebooks (`*.ipynb`), supporting scripts (`*.py`), reports (`lab_report.md`), and when applicable datasets or saved models (`*.h5`, `*.keras`).

## Quick Start
```powershell
# Clone
git clone https://github.com/AmanTewariSkoolKid/deep-learning-lab.git
cd deep-learning-lab

# (Optional) Install Git LFS BEFORE pulling large model files
# choco install git-lfs -y ; git lfs install

# Create virtual environment (Python 3.10+ recommended)
python -m venv .venv ; .\.venv\Scripts\Activate.ps1

# Install aggregated dependencies
pip install -r requirements.txt

# Launch Jupyter
pip install jupyter
jupyter notebook
```
Open a specific notebook (e.g. `ex10/ex10.ipynb`) to explore models.

## Dependencies
Aggregated (union) of exercise requirements:
```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
```
Pin versions for reproducibility if needed (e.g. in a future `requirements.lock`).

## Data & Models
- Large binary model files (`*.h5`, `*.keras`) tracked with Git LFS (recommended). See `.gitattributes`.
- Raw datasets reside inside exercise folders (e.g. `ex5/data/flower_photos`). Avoid committing extremely large, reproducible datasets; prefer download scripts.

## Recommended Practices Implemented
- Root `README.md` with structure & setup.
- Consolidated `requirements.txt` for unified environment.
- `.gitignore` filtering Python, notebook, environment, and temp artifacts.
- `.gitattributes` prepared for Git LFS model tracking.
- `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` for collaboration standards.
- `CHANGELOG.md` initialized (Keep a Changelog format).
- `pre-commit` configuration for consistent formatting & linting.

## Running Individual Exercises
Most notebooks are self-contained. For script-based training (e.g. transfer learning in `ex5`):
```powershell
.\.venv\Scripts\Activate.ps1
python ex5\download_dataset.py   # if dataset download script is provided
python ex5\train_transfer.py     # trains transfer model
```
UNet segmentation (ex10) notebook expects annotation directory structure under `ex10/ann_dir` & images under `ex10/img_dir`.

## Contributing
See `CONTRIBUTING.md`. Open issues for:
- Enhancements (refactoring notebooks into modular packages)
- Adding tests (pytest for utility functions)
- Dataset optimization or reproducibility improvements

## License
This project is proposed under the MIT License (see `LICENSE`). Confirm if an alternative is preferred.

## Code Style & Quality
- Black + isort enforced via `pre-commit`.
- Flake8 ensures lightweight linting.
- Prefer modular refactors: move reusable code from notebooks into `.py` modules.

## Roadmap (Suggested Improvements)
- Add automated tests for utility scripts.
- Introduce experiment tracking (MLflow) for model runs.
- Add version pinning + lock file for reproducibility.
- Containerize environment with `Dockerfile`.
- CI workflow (GitHub Actions) for lint + minimal smoke tests.

## Notebook PDF Export
Export all notebooks to individual PDFs and merge into one consolidated lab report.

```powershell
# Activate environment
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-export.txt  # nbconvert[webpdf], PyPDF2

# Run export script
python scripts\export_notebooks.py --source . --pattern "ex*/**/*.ipynb" --export-dir exports --combined-pdf combined_lab_reports.pdf

# Result:
# - Individual PDFs in .\exports\<notebook>.pdf
# - Merged PDF: .\exports\combined_lab_reports.pdf
```

If Chromium auto-download fails, install Google Chrome or Edge. Fallback LaTeX export requires a TeX distribution (e.g. MiKTeX).

## Acknowledgements
Educational lab work derived from course curriculum; datasets from public sources (e.g., TensorFlow Flowers). See dataset-specific `LICENSE.txt` where provided.
