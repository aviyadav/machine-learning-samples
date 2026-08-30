import numpy as np
import pandas as pd
from tabfm import TabFMRegressor

# Choose your backend:

# OPTION A: JAX Backend
# from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
# model = tabfm_v1_0_0.load(model_type="regression")

# OPTION B: PyTorch Backend
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="regression")

# Initialize scikit-learn compatible regressor (works with either backend model)
reg = TabFMRegressor(model=model)

# Prepare your dataset
# X_train = pd.DataFrame({
#     "sqft": [1200, 2500, 1500, 3000],
#     "neighborhood": ["A", "B", "A", "C"]
# }).convert_dtypes(dtype_backend="pyarrow")
# y_train = np.array([250000, 550000, 310000, 620000])

# Generate a large random dataset (50 entries)
rng = np.random.default_rng(42)
n_samples = 50
X_train = pd.DataFrame({
    "sqft": rng.integers(800, 4000, n_samples),
    "neighborhood": rng.choice(["A", "B", "C"], n_samples)
}).convert_dtypes(dtype_backend="pyarrow")
y_train = rng.integers(150000, 800000, n_samples)

X_test = pd.DataFrame({
    "sqft": [1800, 2800, 3500, 4000],
    "neighborhood": ["A", "B", "C", "A"]
}).convert_dtypes(dtype_backend="pyarrow")

# Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
