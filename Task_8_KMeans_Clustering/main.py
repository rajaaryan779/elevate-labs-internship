### IMPORT REQUIRED LIBRARIES ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

### LOAD DATASET ###
df = pd.read_csv("Mall Customers.csv")

### ENCODE GENDER ###
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

### SELECT FEATURES ###
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

### STANDARDIZE FEATURES ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### PCA FOR 2D VISUALIZATION ###
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

### 2D PLOT OF DATA ###
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50)
plt.title("2D Visualization of Customers (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

### ELBOW METHOD TO FIND OPTIMAL K ###
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.xticks(K_range)
plt.grid(True)
plt.show()

### FIT K-MEANS WITH OPTIMAL K=5 ###
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

### CLUSTER VISUALIZATION ###
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', s=60)
plt.title("Customer Segments (K-Means Clustering with K=5)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

### EVALUATE USING SILHOUETTE SCORE ###
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", round(score, 4))
