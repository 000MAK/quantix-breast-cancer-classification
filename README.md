# Quantix Breast Cancer Classification

Train, validation, and test pipeline for a small binary classification demo using logistic regression and the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.

> **Educational use only.** This model is not a medical device, does not provide a diagnosis, and must not be used to make clinical decisions.

## Dataset

The bundled dataset contains 569 samples, 30 numeric features, and two classes. The project maps **malignant** to the positive class (`1`) and **benign** to `0`. The scikit-learn loader supplies the dataset locally, so the training command does not download data.

Dataset documentation: [scikit-learn `load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

## Setup

From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Pipeline

- Stratified, reproducible split: 64% train, 16% validation, 20% test.
- Standardization and logistic regression are combined in a scikit-learn pipeline to prevent preprocessing leakage.
- Validation and held-out test metrics include precision, recall, F1, ROC-AUC, and confusion matrices.
- The final estimator and JSON metrics are written to `outputs/`.

Use a different output folder or seed with:

```powershell
python train.py --output-dir outputs\run-2 --seed 17
```

The held-out test set is evaluated once and is not used to fit or select the model.

## Results

Local run with the defaults (seed 42), held-out test set of 114 samples: accuracy **0.982**, malignant precision **0.976**, recall **0.976**, ROC-AUC **0.996**. A small, well-studied benchmark; these numbers say nothing about clinical performance.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |
| `train.py` | Training and evaluation script |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
