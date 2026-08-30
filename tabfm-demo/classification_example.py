import numpy as np
import pandas as pd
from tabfm import TabFMClassifier

# Choose your backend:

# OPTION A: JAX Backend
# from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load()

# Initialize scikit-learn compatible classifier (works with either backend model)
clf = TabFMClassifier(model=model)

# Prepare your dataset (supports mixed numerical and categorical features)
# X_train = pd.DataFrame({
#     "age": [25.0, 45.0, 35.0, 50.0],
#     "job": ["engineer", "manager", "engineer", "manager"],
#     "income": [80000, 120000, 90000, 130000]
# }).convert_dtypes(dtype_backend="pyarrow")
# y_train = np.array(["low_risk", "high_risk", "low_risk", "high_risk"])

# Generate a large random dataset (50 entries)
rng = np.random.default_rng(42)
n_samples = 50
X_train = pd.DataFrame({
    "age": rng.uniform(20.0, 65.0, n_samples),
    "job": rng.choice(["engineer", "manager", "technician", "analyst"], n_samples),
    "income": rng.integers(40000, 150000, n_samples)
}).convert_dtypes(dtype_backend="pyarrow")
y_train = rng.choice(["low_risk", "high_risk"], n_samples)

X_test = pd.DataFrame({
    "age": [30.0, 48.0, 55.0, 24.0],
    "job": ["engineer", "manager", "technician", "analyst"],
    "income": [85000, 125000, 100000, 70000]
}).convert_dtypes(dtype_backend="pyarrow")

# Fit classifier (prepares ordinal encoders and numerical scalers)
clf.fit(X_train, y_train)

# Predict classes and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Class Probabilities:\n", probabilities)
