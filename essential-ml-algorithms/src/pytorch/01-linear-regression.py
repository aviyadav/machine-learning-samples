from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from execution_profiler import profile_execution


def _train_test_split(X, y, test_ratio=0.2, seed=42):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(X.size(0), generator=g)
    split = int(X.size(0) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@profile_execution
def test_linear_regression():
    # Predicting house prices from square footage.
    X = torch.tensor([[750.0], [1000.0], [1200.0], [1500.0], [1800.0], [2200.0]])
    y = torch.tensor([[150000.0], [200000.0], [230000.0], [290000.0], [340000.0], [410000.0]])

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_ratio=0.2, seed=42)

    # Scale features/targets for stable optimization, then invert prediction scale.
    x_mean, x_std = X_train.mean(), X_train.std().clamp_min(1e-6)
    y_mean, y_std = y_train.mean(), y_train.std().clamp_min(1e-6)
    X_train_s = (X_train - x_mean) / x_std
    X_test_s = (X_test - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = torch.nn.MSELoss()

    for _ in range(800):
        optimizer.zero_grad()
        pred = model(X_train_s)
        loss = criterion(pred, y_train_s)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(X_test_s) * y_std + y_mean
        mse = torch.mean((predictions - y_test) ** 2).item()

    print(f"Predictions: {predictions.squeeze(-1).tolist()}")
    print(f"MSE: {mse:.2f}")


if __name__ == "__main__":
    test_linear_regression()
