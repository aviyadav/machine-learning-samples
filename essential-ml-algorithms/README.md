# Essential ML Algorithms

A compact, hands-on Python project that demonstrates seven core machine learning algorithms in two implementations:
- scikit-learn based examples in `src/sklearn`
- PyTorch based examples in `src/pytorch`

Each script is intentionally focused on one algorithm and includes execution profiling via `execution_profiler.py`.

## Project Details

- Name: `essential-ml-algorithms`
- Version: `0.1.0`
- Python: `>=3.14`
- Key dependencies:
  - `scikit-learn`
   - `torch`
  - `xgboost`
  - `numpy`
  - `pandas`
  - `matplotlib`

## Included Algorithms (01 to 07)

Each algorithm is available in both folders below:
- `src/sklearn`
- `src/pytorch`

Reference sequence:

1. `01-linear-regression.py`
2. `02-logistic-regression.py`
3. `03-decision-trees-random-forest.py`
4. `04-xgboost.py`
5. `05-k-means-clustering.py`
6. `06-support-vector-machine.py`
7. `07-neural-networks.py`

### sklearn examples

1. `src/sklearn/01-linear-regression.py`
   - Linear regression on a simple house price example
   - Metric: Mean Squared Error (MSE)
2. `src/sklearn/02-logistic-regression.py`
   - Logistic regression on Breast Cancer dataset
   - Metrics: Accuracy + classification report
3. `src/sklearn/03-decision-trees-random-forest.py`
   - Random Forest classification on Iris dataset
   - Outputs accuracy and feature importance
4. `src/sklearn/04-xgboost.py`
   - XGBoost regressor on Diabetes dataset
   - Metric: RMSE
5. `src/sklearn/05-k-means-clustering.py`
   - K-Means clustering for synthetic customer segments
   - Saves visualization as `customer_segments.png`
6. `src/sklearn/06-support-vector-machine.py`
   - SVM (RBF kernel) on Digits dataset
   - Metric: Accuracy
7. `src/sklearn/07-neural-networks.py`
   - MLP neural network on Digits dataset
   - Metric: Accuracy

### PyTorch examples

1. `src/pytorch/01-linear-regression.py`
   - Linear regression implemented with torch modules and optimizers
2. `src/pytorch/02-logistic-regression.py`
   - Binary classifier using `BCEWithLogitsLoss`
3. `src/pytorch/03-decision-trees-random-forest.py`
   - Closest PyTorch supervised analogue for tree/forest workflow
4. `src/pytorch/04-xgboost.py`
   - Closest PyTorch nonlinear regressor analogue for boosted workflow
5. `src/pytorch/05-k-means-clustering.py`
   - K-Means implemented directly with tensor ops
6. `src/pytorch/06-support-vector-machine.py`
   - Linear multiclass SVM-style training with hinge loss
7. `src/pytorch/07-neural-networks.py`
   - MLP classifier using PyTorch

## Execution Profiling

All algorithm scripts use the shared decorator from `execution_profiler.py`.

For each run, profiling prints:
- Elapsed wall-clock time
- CPU time
- Processors available to the process
- Peak Python memory (tracemalloc)
- RSS memory before/after (when `psutil` is installed)
- GPU availability (via `nvidia-smi`)
- Whether GPU was used by the method

## Run Options

### Run a Single Script (sklearn)

```bash
python src/sklearn/01-linear-regression.py
```

### Run a Single Script (PyTorch)

```bash
python src/pytorch/01-linear-regression.py
```

### Run All Scripts (01 → 07, sklearn)

Use the wrapper:

```bash
python src/sklearn/run_all_algos_ex.py
```

### Run All Scripts (01 → 07, PyTorch)

Use the PyTorch wrapper:

```bash
python src/pytorch/run_all_algos_ex.py
```

The wrapper executes scripts in numeric order and prints:
- Per-script output
- Per-script runtime and exit code
- A final success/failure summary

## Setup

Install dependencies from the project root using your preferred workflow.

### Option A: with uv

```bash
uv sync
```

### Option B: with pip

```bash
python -m pip install -U pip
python -m pip install scikit-learn torch xgboost matplotlib pandas pyarrow
```

Optional for enhanced memory profiling:

```bash
python -m pip install psutil
```

## Project Structure

```text
essential-ml-algorithms/
├─ execution_profiler.py
├─ src/
│  ├─ sklearn/
│  │  ├─ 01-linear-regression.py
│  │  ├─ 02-logistic-regression.py
│  │  ├─ 03-decision-trees-random-forest.py
│  │  ├─ 04-xgboost.py
│  │  ├─ 05-k-means-clustering.py
│  │  ├─ 06-support-vector-machine.py
│  │  ├─ 07-neural-networks.py
│  │  └─ run_all_algos_ex.py
│  └─ pytorch/
│     ├─ 01-linear-regression.py
│     ├─ 02-logistic-regression.py
│     ├─ 03-decision-trees-random-forest.py
│     ├─ 04-xgboost.py
│     ├─ 05-k-means-clustering.py
│     ├─ 06-support-vector-machine.py
│     ├─ 07-neural-networks.py
│     └─ run_all_algos_ex.py
├─ pyproject.toml
└─ uv.lock
```
