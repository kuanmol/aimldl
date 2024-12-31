import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# elbow method
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# training k-mean model on dataset
kmeans = KMeans(n_clusters=10, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

#grouping cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cluster 5")
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=100, c="pink", label="Cluster 6")
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=100, c="black", label="Cluster 7")
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s=100, c="orange", label="Cluster 8")
plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s=100, c="grey", label="Cluster 9")
plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s=100, c="brown", label="Cluster 10")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="yellow", label="Centroids")
plt.title("Cluster of customer")
plt.xlabel("Annual income")
plt.ylabel("Spending Score")
plt.show()