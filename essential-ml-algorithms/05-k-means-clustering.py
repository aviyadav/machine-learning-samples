from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from execution_profiler import profile_execution

@profile_execution
def k_means_clustering():
    # Simulating customer data: [purchase_frequency, avg_order_value]
    np.random.seed(42)
    customers = np.random.randn(200, 2) * np.array([2, 50]) + np.array([5, 100])
    customers = np.vstack([
        customers,
        np.random.randn(50, 2) * np.array([1, 20]) + np.array([20, 200])  # VIP cluster
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customers)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    plt.scatter(customers[:, 0], customers[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Purchase Frequency')
    plt.ylabel('Average Order Value ($)')
    plt.title('Customer Segments')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig('customer_segments.png', dpi=150)
    plt.show()
    print(f"Cluster centers (scaled):\n{kmeans.cluster_centers_}")


if __name__ == "__main__":
    k_means_clustering()