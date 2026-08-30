# tabfm-demo

Demo project for [TabFM](https://pypi.org/project/tabfm/) — Google's Tabular Foundation Model — showing zero-shot classification and regression on tabular data.

## What is TabFM?

TabFM (Tabular Foundation Model) is Google's pretrained foundation model for tabular data. Instead of training a model from scratch on your dataset, TabFM is pretrained on large-scale synthetic and real-world tabular corpora and performs **in-context learning** at inference time: you provide your training rows as context, and it predicts on new rows zero-shot — no gradient updates, no hyperparameter tuning, no GPU training loop.

Key properties:

- **Zero-shot / few-shot**: learns a new tabular task at inference time from the examples you pass to `fit()`.
- **Mixed feature types**: handles numerical and categorical features natively (no manual encoding pipeline).
- **Scikit-learn compatible API**: `TabFMClassifier` and `TabFMRegressor` expose the familiar `fit` / `predict` / `predict_proba` interface.
- **Multiple backends**: the same model weights can run on a JAX or PyTorch backend.

This makes it a strong baseline for small-to-medium tabular datasets where a gradient-boosted tree or a neural net would require careful tuning.

## Project contents

| File | Description |
| --- | --- |
| `classification_example.py` | Binary classification (credit-risk style) with `TabFMClassifier` |
| `regression_example.py` | Price prediction with `TabFMRegressor` |
| `multiclass_example.py` | Complex 5-class classification on a larger, mixed-type dataset using the `ensemble` preset |
| `pyproject.toml` | Project metadata and dependencies (managed with `uv`) |

Both examples:

- Build a randomly generated 50-entry training set (`X_train` / `y_train`) with a fixed seed for reproducibility. The original small 4-row datasets are kept as comments.
- Use **pandas with the pyarrow backend** (`convert_dtypes(dtype_backend="pyarrow")`), so columns are stored as `double[pyarrow]` / `string[pyarrow]` rather than numpy/object dtypes.
- Use the PyTorch backend (`tabfm_v1_0_0_pytorch`); the JAX backend is available as a commented-out alternative.

## Requirements

- Python >= 3.13
- Dependencies: `tabfm`, `pandas`, `pyarrow`, `torch`, `safetensors`, `polars`

## Setup

```sh
uv sync
```

This creates a virtual environment and installs all dependencies from `pyproject.toml` / `uv.lock`.

## Usage

Run the classification example:

```sh
.venv/bin/python classification_example.py
```

Run the regression example:

```sh
.venv/bin/python regression_example.py
```

Run the multiclass example:

```sh
.venv/bin/python multiclass_example.py
```

On first run, the pretrained model weights are downloaded automatically (via Hugging Face) and cached locally.

### Expected output

Classification (predicts risk class and class probabilities):

```
Predictions: ['high_risk' 'high_risk']
Class Probabilities:
 [[0.570996   0.42900407]
 [0.6364284  0.36357155]]
```

Regression (predicts prices for the held-out rows):

```
Predicted Prices: [464788.47 481240.  ]
```

Multiclass (fits a 5-class loyalty-tier model and prints accuracy + a classification report):

```
Train: 306 rows, Test: 54 rows
Classes: ['bronze', 'silver', 'gold', 'platinum', 'diamond']

Accuracy: 0.963
```

(Exact numbers depend on the random dataset and model weights.)

## How it works

1. **Load the model** — `tabfm_v1_0_0.load()` downloads/loads the pretrained weights.
2. **Wrap it** — `TabFMClassifier` / `TabFMRegressor` provide the scikit-learn interface.
3. **Fit** — `clf.fit(X_train, y_train)` prepares ordinal encoders for categorical features and scalers for numerical features, and stores the training rows as in-context examples.
4. **Predict** — `clf.predict(X_test)` / `reg.predict(X_test)` run the foundation model zero-shot over the test rows.

## Notes

- Since the demo DataFrames are built in memory, the pyarrow integration uses `dtype_backend="pyarrow"`. When loading data from CSV/Parquet files, you can additionally pass `engine="pyarrow"` to `pd.read_csv` / `pd.read_parquet`.
- To switch to the JAX backend, swap the import in either example:

  ```python
  from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
  ```
