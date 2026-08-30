"""Complex multiclass example for the Tabular Foundation Model (TabFM).

Demonstrates a realistic, larger dataset with mixed feature types and a
multi-class target (5 customer "loyalty tiers"), using TabFM's heavier
`ensemble` preset (feature crosses + SVD features + NNLS-weighted blending +
per-problem calibration) and a proper train/test evaluation.

Unlike `classification_example.py` / `regression_example.py`, this example:
  * uses ~300 training rows (and a held-out test set),
  * mixes numeric, categorical, boolean, and datetime features,
  * solves a 5-class problem (TabFM supports up to 10 classes),
  * evaluates with sklearn metrics rather than just printing predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tabfm import TabFMClassifier

# Choose your backend:

# OPTION A: JAX Backend
# from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
# model = tabfm_v1_0_0.load()

# OPTION B: PyTorch Backend
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

model = tabfm_v1_0_0.load()

# ---------------------------------------------------------------------------
# Build a synthetic "customer loyalty tier" dataset.
#
# We generate features with a seeded RNG so the example is reproducible, and
# derive the 5-class target from a simple value score so there is real signal
# for the model to pick up (rather than purely random labels).
# ---------------------------------------------------------------------------

TIERS = ["bronze", "silver", "gold", "platinum", "diamond"]


def make_dataset(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.normal(40.0, 12.0, n).clip(18.0, 80.0)
    tenure_months = rng.integers(0, 121, n)
    monthly_spend = rng.uniform(5.0, 200.0, n).round(2)
    num_logins = rng.integers(0, 301, n)
    num_support_tickets = rng.poisson(2.0, n)
    signup_date = pd.to_datetime(
        "2021-01-01"
    ) + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D")

    # A few high-cardinality categorical features.
    region = rng.choice(
        ["north", "south", "east", "west", "central", "coastal"], n
    )
    device_type = rng.choice(
        ["mobile", "desktop", "tablet", "smart_tv"], n
    )
    plan_type = rng.choice(
        ["free", "basic", "standard", "premium"], n
    )
    has_referral = rng.choice([True, False], n)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "num_logins": num_logins,
            "num_support_tickets": num_support_tickets,
            "region": region,
            "device_type": device_type,
            "plan_type": plan_type,
            "has_referral": has_referral,
            "signup_date": signup_date,
        }
    )


def make_target(df: pd.DataFrame) -> np.ndarray:
    """Map a simple 'value' score to one of five loyalty tiers."""
    score = (
        0.4 * (df["monthly_spend"] / 200.0)
        + 0.3 * (df["tenure_months"] / 120.0)
        + 0.2 * (df["num_logins"] / 300.0)
        - 0.1 * (df["num_support_tickets"] / 8.0)
    )
    tier_idx = np.digitize(score, bins=[0.20, 0.35, 0.50, 0.65])
    return np.array(TIERS)[tier_idx]


rng = np.random.default_rng(42)
data = make_dataset(rng, 360)
y = make_target(data)

# Use pyarrow-backed dtypes for numeric/string columns (same pattern as the
# other examples). The datetime column stays as numpy datetime64 so TabFM's
# datetime transformer recognizes it.
X = data.convert_dtypes(dtype_backend="pyarrow")
X["signup_date"] = data["signup_date"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
print(f"Classes: {TIERS}\n")

# ---------------------------------------------------------------------------
# Fit a TabFMClassifier using the "ensemble" preset (heavier ensembling +
# calibration), with a reduced estimator count to keep runtime modest.
# ---------------------------------------------------------------------------
clf = TabFMClassifier.ensemble(
    model,
    n_estimators=16,
    random_state=0,
)

clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluate on the held-out test set.
# ---------------------------------------------------------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=TIERS,
        target_names=TIERS,
        zero_division=0,
    )
)

# Show a few example predictions with their confidence.
sample_idx = np.argsort(-y_proba.max(axis=1))[:5]
print("Most confident test predictions:")
for i in sample_idx:
    top = int(np.argmax(y_proba[i]))
    print(
        f"  true={y_test[i]:<8} pred={clf.classes_[top]:<8} "
        f"conf={y_proba[i][top]:.3f}"
    )
