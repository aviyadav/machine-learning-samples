from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from execution_profiler import profile_execution


def _kmeans_torch(X, n_clusters=3, max_iters=100, seed=42):
    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(X.size(0), generator=g)[:n_clusters]
    centers = X[init_idx].clone()

    for _ in range(max_iters):
        distances = torch.cdist(X, centers)
        labels = distances.argmin(dim=1)

        new_centers = []
        for c in range(n_clusters):
            members = X[labels == c]
            if members.numel() == 0:
                new_centers.append(centers[c])
            else:
                new_centers.append(members.mean(dim=0))
        new_centers = torch.stack(new_centers)

        if torch.allclose(centers, new_centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return labels, centers


@profile_execution
def k_means_clustering():
    # Simulating customer data: [purchase_frequency, avg_order_value].
    torch.manual_seed(42)
    customers = torch.randn(200, 2) * torch.tensor([2.0, 50.0]) + torch.tensor([5.0, 100.0])
    vip_cluster = torch.randn(50, 2) * torch.tensor([1.0, 20.0]) + torch.tensor([20.0, 200.0])
    customers = torch.cat([customers, vip_cluster], dim=0)

    mean, std = customers.mean(dim=0), customers.std(dim=0).clamp_min(1e-6)
    X_scaled = (customers - mean) / std

    labels, centers = _kmeans_torch(X_scaled, n_clusters=3, max_iters=100, seed=42)

    plt.scatter(
        customers[:, 0].numpy(),
        customers[:, 1].numpy(),
        c=labels.numpy(),
        cmap="viridis",
        alpha=0.6,
    )
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Average Order Value ($)")
    plt.title("Customer Segments (PyTorch K-Means)")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig("customer_segments_torch.png", dpi=150)
    plt.show()

    print(f"Cluster centers (scaled):\n{centers}")


if __name__ == "__main__":
    k_means_clustering()
