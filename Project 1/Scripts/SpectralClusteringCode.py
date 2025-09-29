import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#Load and Clean Data 
mall_data = pd.read_csv('../Mall_Customers.csv')
mall_data = pd.get_dummies(mall_data, drop_first=True)
mall_data = mall_data.drop('CustomerID', axis = 1)
mall_data = mall_data.ffill()

mall_data.head()

#Data visualization
scaler = StandardScaler()
mall_data_scaled = scaler.fit_transform(mall_data)
mall_data_normalized = normalize(mall_data_scaled)
mall_data_normalized = pd.DataFrame(mall_data_normalized)
pca = PCA(n_components = 2)
mall_data_principal = pca.fit_transform(mall_data_normalized)
mall_data_principal = pd.DataFrame(mall_data_principal)
mall_data_principal.columns = ['P1', 'P2']

mall_data_principal.head()

#Building Clustering models
spectral_model_rbf = SpectralClustering(n_clusters = 3, affinity = 'rbf')
labels_rbf = spectral_model_rbf.fit_predict(mall_data_principal)
colors = {0: 'red', 1: 'green', 2: 'yellow'}
cvec = [colors[label] for label in labels_rbf]



plt.figure(figsize =(8,8))
plt.scatter(mall_data_principal['P1'], mall_data_principal['P2'], c = labels_rbf, cmap='viridis', s=50)
plt.xlabel('P1')
plt.ylabel('P2')
plt.title('Spectral Clustering (RBF)')
#plt.colorbar(label="Cluster")
plt.show()

#Nearest neighbors
spectral_model_nn = SpectralClustering(n_clusters = 3, affinity = 'nearest_neighbors')
labels_nn = spectral_model_nn.fit_predict(mall_data_principal)

#Evaluating performance
affinity = ['rbf', 'nearest_neighbors']
s_scores = [silhouette_score(mall_data, labels_rbf), silhouette_score(mall_data, labels_nn)]

print(s_scores)

#Comparing performance
plt.bar(affinity, s_scores)
plt.xlabel('Affinity')
plt.ylabel('Silhouette Score')
plt.title('Comparison of different Clustering Models')
plt.show()