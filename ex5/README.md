# Transfer Learning Experiment (Experiment 5)

Implements image classification with transfer learning using a JSON configuration.

## Files
- `config_transfer.json` : Configuration for dataset, model, training, augmentation, outputs.
- `train_transfer.py` : Script that reads the config and runs training + (optional) fine-tuning + evaluation.
- `note.md` : Theory notes and rationale.

## Expected Dataset Structure
```
ex5/
  data/
    flowers/
      class1/
        img1.jpg
        img2.jpg
      class2/
        ...
      ...
```
Update `dataset.path` in the config if your path differs.

If the dataset folder does not exist, the batch script will attempt to download the TensorFlow flower photos archive automatically (URL defined in `config_transfer.json` under `dataset.download_url`).

## Running
Activate your virtual environment first, then install dependencies if needed:
```
pip install tensorflow numpy scikit-learn matplotlib
```
Run training:
```
python train_transfer.py

Or use the automated setup (creates venv, installs deps, downloads dataset if missing, runs training):
```
setup_and_run.bat
```
```

## Configuration Highlights
- `trainable_strategy`: `freeze_then_finetune` first trains new head, then unfreezes last N layers.
- `fine_tune_unfreeze_layers`: how many final layers of the base model to unfreeze.
- `augmentation` section toggles common data augmentation.
- `outputs` controls which evaluation artifacts are printed/saved.

## Extending
- Add more metrics by modifying `metrics` array.
- Change `base_architecture` to another Keras applications model (e.g., `ResNet50`).
- Adjust image size to match architecture requirements.

## Notes
If you change number of classes, update `head.output_classes`.
