# Brain Tumor Classification with CNN + Radiomics Fusion

This repository contains an end-to-end research pipeline for MRI brain tumor classification for CSC4810 Artificial Intelligence Final Project that combines a pretrained CNN (MobileNetV2) with engineered radiomics features and demonstrates interpretability via Grad-CAM. The project includes data preparation, model training, evaluation (accuracy & macro-F1), robustness experiments (noise / resolution), and visualization of Grad-CAM overlays.

---

## Key Features

- Multimodal fusion: combine CNN embeddings (learned) with radiomics (hand-crafted) features.
- Lightweight backbone: MobileNetV2 pretrained on ImageNet with a custom head.
- Interpretability: Grad-CAM implementation & visualization for qualitative model inspection.
- Robustness experiments: tests across noise levels and image resolutions.
- Reproducible notebooks and helper utilities for training, evaluation, and visualization.

---

## Repository Structure

- `brain_tumor_mini.ipynb` / `cnn_baseline_gradcam.ipynb` — primary Jupyter notebook(s) containing the full pipeline: data loading, model building/training, radiomics pipeline, fusion classifier, experiments, and Grad-CAM visualization.
- `data/` (optional) — place the downloaded dataset here or follow the instructions below to download it via KaggleHub or Kaggle CLI.
- `notebooks/` — extra exploratory notebooks (if any).
- `models/` — recommended location to save checkpointed models (`.pt`) and sklearn pipelines (`.pkl` / `.joblib`).
- `results/` — suggested folder for saving evaluation outputs, plots, and CSVs.
- `.gitignore` — recommended ignores (check for model checkpoints, dataset, large outputs).

---

## Dataset

This project uses the Kaggle dataset: **sartajbhuvaji/brain-tumor-classification-mri** (organized as `Training` and `Testing` directories).

Options to obtain the data:

1. KaggleHub (used in the notebook): the notebooks call `kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")` to fetch and unpack the dataset automatically.
2. Kaggle CLI: install the Kaggle CLI, create an API token, then:

```powershell
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p data --unzip
```

3. Manual: download from the dataset page and extract into `data/` with top-level folders `Training/` and `Testing/`.

After downloading, ensure the notebook's `path` variable points to the dataset root.

---

## Requirements

Suggested Python packages (create a virtual environment and install these):

```text
python>=3.8
torch
torchvision
numpy
scikit-learn
scikit-image
opencv-python
matplotlib
pandas
kagglehub
joblib
pyradiomics  # optional, if radiomics are computed using pyradiomics

# For notebook usage
jupyterlab

```

You can create a `requirements.txt` and install via:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: If you have a CUDA-capable GPU and want to use it, install a matching `torch` wheel (see https://pytorch.org/get-started/locally/).

---

## Running the Notebook

1. Open `cnn_baseline_gradcam.ipynb` (or `brain_tumor_mini.ipynb`) in Jupyter or VS Code Notebook.
2. Ensure the `path` variable points to the dataset root (or run the KaggleHub cell to download it).
3. Run cells in order. Key sections:
   - Environment & imports (verify package versions)
   - Data transforms and `ImageFolder` creation
   - Train / validation split
   - Model build (MobileNetV2) and training loop
   - Radiomics extraction & classical ML pipeline
   - Feature fusion and evaluation
   - Grad-CAM generation and visualization

Example: open a terminal and run (Windows PowerShell):

```powershell
jupyter lab
# or
jupyter notebook
```

---

## Running Training (script-style)

The notebooks implement an interactive workflow. If you extract key steps into scripts, the high-level commands are:

```powershell
# 1. Prepare dataset path and config
# 2. Train CNN backbone (or fine-tune)
python train_cnn.py --data-dir data --epochs 10 --batch-size 16 --lr 1e-4

# 3. Extract CNN embeddings and compute radiomics features
python extract_features.py --data-dir data --out-dir features

# 4. Train fusion classifier (sklearn)
python train_fusion.py --features features --out models/fusion.joblib

# 5. Evaluate and visualize Grad-CAM
python evaluate_and_gradcam.py --model models/cnn.pt --data-dir data --out results/
```


---

## Reproducibility Tips

- Set seeds at the top of notebooks: `random.seed(...)`, `np.random.seed(...)`, `torch.manual_seed(...)`.
- Record package versions (a cell in the notebook already prints versions). Consider saving `pip freeze > requirements-freeze.txt`.
- Save model checkpoints (use `torch.save(model.state_dict(), "models/model.pt")`) and sklearn pipelines with `joblib.dump(pipeline, "models/pipeline.joblib")`.
- If using GPU, note exact CUDA and cuDNN versions in README or `requirements-freeze.txt`.

---

## Experiments & Results

- The notebooks contain robustness experiments that add Gaussian noise and change resolution to quantify performance degradation for: CNN-only, radiomics-only, and fused models.
- Metrics tracked: accuracy and macro F1. Per-class metrics and confusion matrices are also computed in evaluation cells.
- Visual outputs: Grad-CAM overlays and comparative plots of accuracy vs noise/resolution.


---

## Grad-CAM Interpretability 

- The notebook defines `generate_gradcam(model, image_tensor, target_class=None)` that:
  - Hooks the last convolutional block in MobileNetV2, captures activations and gradients, computes weights via global pooling of gradients, forms CAM, applies ReLU and normalization, and returns a heatmap numpy array.
- Example visualization is included in the notebook: original image, heatmap, and overlay.

---


## License

This project does not contain a license file by default. If you want to open-source it, consider adding an `MIT` or `Apache-2.0` license. Add `LICENSE` to the repository root.

---

## Contact

Author: Areeb Ehsan
