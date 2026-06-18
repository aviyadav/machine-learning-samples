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
def test_xgboost():
    # Gradient-boosted trees are not native in PyTorch; use a compact nonlinear regressor.
    torch.manual_seed(42)
    X = torch.randn(600, 10)
    y = (
        20 * X[:, 0]
        - 15 * X[:, 1]
        + 8 * X[:, 2] * X[:, 3]
        + 10 * torch.sin(X[:, 4])
        + 2.5 * torch.randn(600)
    ).unsqueeze(1)

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_ratio=0.2, seed=42)

    mean, std = X_train.mean(0, keepdim=True), X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(350):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(X_test)
        rmse = torch.sqrt(torch.mean((predictions - y_test) ** 2)).item()

    print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    test_xgboost()
