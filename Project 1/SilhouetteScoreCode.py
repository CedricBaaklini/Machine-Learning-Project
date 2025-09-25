import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

mall_customer_data = pd.read_csv("Mall_Customers.csv")
mall_customer_data = pd.get_dummies(mall_customer_data, drop_first=True)
mall_customer_data = mall_customer_data.ffill()

print("Data, shape:", mall_customer_data.shape)

#Spectral clustering
model = SpectralClustering(n_clusters=3, affinity='rbf', gamma=0.01)
labels = model.fit_predict(mall_customer_data)

#Silhouette score
score = silhouette_score(mall_customer_data, labels)
print("Silhouette Score:", score)
