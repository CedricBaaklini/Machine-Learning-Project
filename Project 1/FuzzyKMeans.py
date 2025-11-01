# --- Imports: Data handling, clustering, scaling, plotting ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# --- Fuzzy Silhouette Score Function ---
# Calculates the fuzzy silhouette score for a given membership matrix u and data X.
def fuzzy_silhouette(X, u):
    n_samples = X.shape[0]
    n_clusters = u.shape[0]
    dists = pairwise_distances(X)
    sil_scores = []
    for i in range(n_samples):
        a = 0
        b = np.inf
        for k in range(n_clusters):
            weights = u[k]
            a_k = np.sum(weights * dists[i]) / np.sum(weights)
            if u[k, i] == np.max(u[:, i]):
                a = a_k
            else:
                b = min(b, a_k)
        sil = (b - a) / max(a, b)
        sil_scores.append(sil)
    return np.mean(sil_scores)

# --- Membership Table Output Function ---
# Saves the first and last 10 customers' cluster memberships to CSV for review.
def show_memberships(df, u, hard_labels, method_name, filename_prefix):
    k = u.shape[0]
    # First 10 customers
    data_first = []
    for i in range(10):
        row = {
            "Income": df.iloc[i]['Annual Income (k$)'],
            "Spending Score": df.iloc[i]['Spending Score (1-100)'],
            "Assigned Cluster": hard_labels[i]
        }
        for c in range(k):
            row[f"Cluster {c} Membership"] = round(u[c, i], 3)
        data_first.append(row)
    # Last 10 customers
    data_last = []
    for i in range(-10, 0):
        row = {
            "Income": df.iloc[i]['Annual Income (k$)'],
            "Spending Score": df.iloc[i]['Spending Score (1-100)'],
            "Assigned Cluster": hard_labels[i]
        }
        for c in range(k):
            row[f"Cluster {c} Membership"] = round(u[c, i], 3)
        data_last.append(row)
    # Save to CSV for batch review
    pd.DataFrame(data_first).to_csv(f"{filename_prefix}_first10.csv", index=False)
    pd.DataFrame(data_last).to_csv(f"{filename_prefix}_last10.csv", index=False)

# --- Main Analysis Pipeline ---
def main():
    # Load data
    df = pd.read_csv('Mall_Customers.csv')
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head().to_string(index=False))


    
    print("Dataset info:")
    print(df.info())

    # Feature selection and scaling
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using FPC and fuzzy silhouette
    K = range(2, 11)
    fpcs = []
    silhouette_scores = []
    np.random.seed(42)
    for k in K:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, k, 2, error=0.005, maxiter=1000, init=None
        )
        fpcs.append(fpc)
        score = fuzzy_silhouette(X_scaled, u)
        silhouette_scores.append(score)

    # Plot silhouette scores vs k
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'go-', linewidth=2)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Fuzzy Silhouette Score')
    plt.title('Fuzzy Silhouette Score vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig('silhouette_vs_k.png')
    plt.close()

    # Determine optimal k by silhouette
    optimal_k_sil = K[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters by fuzzy silhouette: {optimal_k_sil}")

    # Plot FPC vs k
    plt.figure(figsize=(10, 6))
    plt.plot(K, fpcs, 'bo-', linewidth=2)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Fuzzy Partition Coefficient (FPC)')
    plt.title('FPC vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.savefig('fpc_vs_k.png')
    plt.close()

    # Determine optimal k by FPC
    optimal_k = K[fpcs.index(max(fpcs))]
    print(f"Optimal number of clusters: {optimal_k}")

    # --- FPC-optimal clustering ---
    cntr_fpc, u_fpc, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_scaled.T, optimal_k, 2, error=0.005, maxiter=1000, init=None
    )
    hard_labels_fpc = np.argmax(u_fpc, axis=0)

    # --- Silhouette-optimal clustering ---
    cntr_sil, u_sil, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_scaled.T, optimal_k_sil, 2, error=0.005, maxiter=1000, init=None
    )
    hard_labels_sil = np.argmax(u_sil, axis=0)

    # --- Plot FPC-optimal clustering ---
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_points = X_scaled[hard_labels_fpc == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i], label=f'Cluster {i}', alpha=0.7, s=40)
    plt.scatter(cntr_fpc[:, 0], cntr_fpc[:, 1], c='black', marker='x', s=200, linewidth=3, label='Centroids')
    plt.xlabel('Annual Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')
    plt.title(f'Fuzzy K-Means Clustering (FPC Optimal, k={optimal_k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fpc_optimal_clustering.png')
    plt.close()

    # --- Fuzziness visualization for FPC-optimal clustering ---
    max_membership = np.max(u_fpc, axis=0)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=max_membership, cmap='viridis', s=40)
    plt.colorbar(scatter, label='Max Membership Value')
    plt.scatter(cntr_fpc[:, 0], cntr_fpc[:, 1], c='red', marker='x', s=200, linewidth=3, label='Centroids')
    plt.xlabel('Annual Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')
    plt.title('Fuzzy K-Means: Fuzziness Visualization (FPC Optimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fpc_fuzziness.png')
    plt.close()

    # --- Plot Silhouette-optimal clustering ---
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k_sil))
    for i in range(optimal_k_sil):
        cluster_points = X_scaled[hard_labels_sil == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i], label=f'Cluster {i}', alpha=0.7, s=40)
    plt.scatter(cntr_sil[:, 0], cntr_sil[:, 1], c='black', marker='x', s=200, linewidth=3, label='Centroids')
    plt.xlabel('Annual Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')
    plt.title(f'Fuzzy K-Means Clustering (Silhouette Optimal, k={optimal_k_sil})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('silhouette_optimal_clustering.png')
    plt.close()

    # --- Save memberships for first and last 10 customers for both methods ---
    show_memberships(df, u_fpc, hard_labels_fpc, "FPC", "fpc")
    show_memberships(df, u_sil, hard_labels_sil, "Silhouette", "silhouette")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()