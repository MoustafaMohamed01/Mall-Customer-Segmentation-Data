import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.style.use("dark_background")
plt.figure(figsize=(8, 5), dpi=150)

plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='#00FFCC', markersize=8, linewidth=2, alpha=0.8)

plt.title("The Elbow Point Graph", fontsize=18, color="#00FFCC", weight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=12, color="#A1A1A1", labelpad=10)
plt.ylabel("WCSS (Within Cluster Sum of Squares)", fontsize=12, color="#A1A1A1", labelpad=10)

plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
clusters = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6), dpi=150)
colors = ["green", "blue", "red", "violet", "yellow"]
labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]

for i in range(0,5):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], s=50, color=colors[i], label=labels[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c="cyan", marker="X", label="Centroids")
    
plt.title("Customer Segmentation")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()