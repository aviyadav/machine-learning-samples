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


def _binary_report(y_true, y_pred):
    y_true = y_true.int()
    y_pred = y_pred.int()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return accuracy, precision, recall, f1


@profile_execution
def logistic_regression():
    # Synthetic binary classification with two clusters.
    torch.manual_seed(42)
    n_per_class = 320

    class0 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([-1.5, -1.0])
    class1 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([1.5, 1.0])
    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)], dim=0).unsqueeze(1)

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_ratio=0.2, seed=42)

    mean, std = X_train.mean(0, keepdim=True), X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = torch.nn.Linear(2, 1)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(300):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test))
        y_pred = (probs >= 0.5).float()
        accuracy, precision, recall, f1 = _binary_report(y_test.squeeze(1), y_pred.squeeze(1))

    print(f"Accuracy: {accuracy:.2%}")
    print("Classification Report:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    logistic_regression()
