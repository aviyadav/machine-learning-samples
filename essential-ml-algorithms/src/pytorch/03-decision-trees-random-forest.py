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
def test_random_forest():
    # Tree/forest are not native in PyTorch; use an MLP classifier for a close supervised analogue.
    torch.manual_seed(42)
    num_samples_per_class = 100

    c0 = torch.randn(num_samples_per_class, 4) * 0.45 + torch.tensor([5.0, 3.5, 1.5, 0.2])
    c1 = torch.randn(num_samples_per_class, 4) * 0.45 + torch.tensor([6.0, 2.8, 4.2, 1.3])
    c2 = torch.randn(num_samples_per_class, 4) * 0.45 + torch.tensor([6.8, 3.0, 5.5, 2.0])

    X = torch.cat([c0, c1, c2], dim=0)
    y = torch.cat([
        torch.zeros(num_samples_per_class, dtype=torch.long),
        torch.ones(num_samples_per_class, dtype=torch.long),
        torch.full((num_samples_per_class,), 2, dtype=torch.long),
    ])

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_ratio=0.2, seed=42)
    mean, std = X_train.mean(0, keepdim=True), X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 3),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for _ in range(280):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    first_layer = model[0].weight.detach().abs().mean(dim=0)
    scores = sorted(zip(feature_names, first_layer.tolist()), key=lambda x: x[1], reverse=True)

    print(f"Accuracy: {accuracy:.2%}")
    print("\nFeature Importance:")
    for name, score in scores:
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    test_random_forest()
