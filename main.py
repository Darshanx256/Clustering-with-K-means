import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load dataset
df = pd.read_csv("Mall_Customers.csv")

# just use the needed columns
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# scale it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# elbow method to find optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, 'bo-')
plt.xlabel('K')
plt.ylabel('Q')
plt.title('Elbow Method')
plt.show()

# the plot diminishes at 3
# optimal K should be 3 as per my research

# choose k (say 5, based on elbow)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)

# plot clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title('Clusters')
plt.show()

# check silhouette score
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)
