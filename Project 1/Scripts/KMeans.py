import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv('../Mall_Customers.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("Dataset info:")
print(df.info())

features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("Features selected for clustering:")
print(features.head())
print("\nFeatures statistics:")
print(features.describe())

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

features_normalized_df = pd.DataFrame(features_normalized, columns = ['Annual Income (normalized)', 'Spending Score (normalized)'])

print("Normalized Features:")
print(features_normalized_df)
print("\nNormalized features statistics:")
print(features_normalized_df.describe())

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(features_normalized)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_normalized, kmeans.labels_))

plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.grid(True)

plt.tight_layout()
plt.show()

for k, score in zip(k_range, silhouette_scores):
    print(f"k = {k}: Silhouette Score = {score:.3f}")

optimal_k = 5
# 5 is the most common and therefore safest number to use as the optimal number of clusters
kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = 10)
cluster_labels = kmeans.fit_predict(features_normalized)

df['Cluster'] = cluster_labels
features_normalized_df['Cluster'] = cluster_labels

print(f"K-means clustering complete with k = {optimal_k}")
print(f"Silhouette Score: {silhouette_score(features_normalized, cluster_labels):.3f}")

print("\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())

plt.figure(figsize = (15, 10))

plt.subplot(2, 2, 1)
colors = ['red', 'green', 'blue', 'orange', 'purple']

for i in range(optimal_k):
    cluster_data = features_normalized_df[features_normalized_df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (normalized)'], cluster_data['Annual Income (normalized)'], c = colors[i], label = f'Cluster {i}', alpha = 0.7)

centers_normalized = kmeans.cluster_centers_
plt.scatter(centers_normalized[:, 0], centers_normalized[:, 1], c = 'black', marker  = 'x', s = 200, linewidths = 3, label = 'Centroids')

plt.xlabel('Annual Income (normalized)')
plt.ylabel('Spending Score (normalized)')
plt.title('K-means Clustering (Normalized Data)')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.subplot(2, 2, 2)

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], c = colors[i], label = f'Cluster {i}', alpha = 0.7)

centers_original = scaler.inverse_transform(centers_normalized)
plt.scatter(centers_original[:, 0], centers_original[:, 1], c = 'black', marker  = 'x', s = 200, linewidths = 3, label = 'Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering (Original Scale)')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.subplot(2, 2, 3)

df.boxplot(column = 'Annual Income (k$)', by = 'Cluster', ax = plt.gca())
plt.title('Annual Income Distribution by Cluster')
plt.suptitle('')

plt.subplot(2, 2, 4)

df.boxplot(column = 'Spending Score (1-100)', by = 'Cluster', ax = plt.gca())
plt.title('Spending Score Distribution by Cluster')
plt.suptitle('')

plt.tight_layout()
plt.show()

print("Cluster Analysis:")
print("=" * 50)

cluster_analysis = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
    'CustomerID': 'count'
}).round(2)

cluster_analysis.columns = ['Income_Mean', 'Income_Std', 'Income_Min', 'Income_Max', 'Spending_Mean', 'Spending_Std', 'Spending_Min', 'Spending_Max', 'Customer_Count']

print(cluster_analysis)
print("\nIncome * 1000")

print("\nCluster Interpretation:")
print("=" * 50)

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    count = len(cluster_data)

    print(f"\nCluster {i} ({count} customers):")
    print(f"  Average Income: ${avg_income:.1f}k")
    print(f"  Average Spending Score: {avg_spending:.1f}")

    if avg_income < 40 and avg_spending < 50:
        interpretation = "Low Income, Low Spending (Careful Customers)"
    elif avg_income < 40 and avg_spending >= 50:
        interpretation = "Low Income, High Spending (Impulsive Customers)"
    elif avg_income >= 70 and avg_spending < 50:
        interpretation = "High Income, Low Spending (Conservative Wealthy Customers)"
    elif avg_income >= 70 and avg_spending >= 50:
        interpretation = "High Income, High Spending (Premium Customers)"
    else:
        interpretation = "Medium Income, Median Spending (Standard Customers)"

    print(f"  Interpretation: {interpretation}")

plt.figure(figsize = (12, 8))

colors = ['red', 'green', 'blue', 'orange', 'purple']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], c = colors[i], label = f'Cluster {i}', alpha = 0.7, s = 50)

centers_original = scaler.inverse_transform(centers_normalized)
plt.scatter(centers_original[:, 0], centers_original[:, 1], c = 'black', marker = 'x', s = 300, linewidths = 3, label = 'Centroids')

for i, (x, y) in enumerate(centers_original):
    plt.annotate(f'C{i}', (x, y), xytext = (5, 5), textcoords = 'offset points', fontweight = 'bold', fontsize = 12)

plt.xlabel('Annual Income (k$)', fontsize = 12)
plt.ylabel('Spending Score (1-100)', fontsize = 12)
plt.title('K-means Clustering: Annual Income vs Spending Score\n(Normalized Data Analysis)', fontsize = 14)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
