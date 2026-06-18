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
def test_neural_network():
    # Digits-like synthetic classification with a 64-feature input.
    torch.manual_seed(42)
    num_classes = 10
    samples_per_class = 180
    num_features = 64

    centers = torch.randn(num_classes, num_features) * 2.5
    X = []
    y = []
    for cls in range(num_classes):
        X.append(centers[cls] + 1.0 * torch.randn(samples_per_class, num_features))
        y.append(torch.full((samples_per_class,), cls, dtype=torch.long))

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_ratio=0.2, seed=42)

    mean, std = X_train.mean(0, keepdim=True), X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Sequential(
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    test_neural_network()
