# 🔍 Clustering: Unsupervised Learning for Data Discovery

> **Complete guide to clustering algorithms and their applications**

## 🎯 **Learning Objectives**

- Master clustering algorithms and their use cases
- Understand distance metrics and similarity measures
- Implement clustering algorithms from scratch
- Apply clustering to real-world problems
- Evaluate clustering performance

## 📚 **Table of Contents**

1. [Clustering Fundamentals](#clustering-fundamentals)
2. [Distance Metrics](#distance-metrics)
3. [Clustering Algorithms](#clustering-algorithms)
4. [Implementation Examples](#implementation-examples)
5. [Applications](#applications)
6. [Interview Questions](#interview-questions)

---

## 🔍 **Clustering Fundamentals**

### **Concept**

Clustering is an unsupervised learning technique that groups similar data points together without prior knowledge of the groups.

### **Key Concepts**

1. **Cluster**: A group of similar data points
2. **Centroid**: The center point of a cluster
3. **Distance Metric**: Measure of similarity between data points
4. **Clustering Algorithm**: Method to group data points

### **Types of Clustering**

- **Partitioning**: K-means, K-medoids
- **Hierarchical**: Agglomerative, Divisive
- **Density-based**: DBSCAN, OPTICS
- **Model-based**: Gaussian Mixture Models
- **Grid-based**: STING, CLIQUE

---

## 📏 **Distance Metrics**

### **1. Euclidean Distance**

**Formula**: `d(x, y) = √(Σ(x_i - y_i)²)`

**Code Example**:
```python
import numpy as np
from scipy.spatial.distance import euclidean

def euclidean_distance(x, y):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x - y) ** 2))

# Example usage
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
distance = euclidean_distance(x, y)
print(f"Euclidean distance: {distance}")

# Using scipy
distance_scipy = euclidean(x, y)
print(f"Scipy distance: {distance_scipy}")
```

### **2. Manhattan Distance**

**Formula**: `d(x, y) = Σ|x_i - y_i|`

**Code Example**:
```python
from scipy.spatial.distance import cityblock

def manhattan_distance(x, y):
    """Calculate Manhattan distance between two points"""
    return np.sum(np.abs(x - y))

# Example usage
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
distance = manhattan_distance(x, y)
print(f"Manhattan distance: {distance}")

# Using scipy
distance_scipy = cityblock(x, y)
print(f"Scipy distance: {distance_scipy}")
```

### **3. Cosine Similarity**

**Formula**: `cos(x, y) = (x · y) / (||x|| × ||y||)`

**Code Example**:
```python
from scipy.spatial.distance import cosine

def cosine_similarity(x, y):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

# Example usage
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
similarity = cosine_similarity(x, y)
print(f"Cosine similarity: {similarity}")

# Using scipy
distance_scipy = cosine(x, y)
print(f"Cosine distance: {distance_scipy}")
```

---

## 🎯 **Clustering Algorithms**

### **1. K-Means Clustering**

**Concept**: Partition data into k clusters by minimizing within-cluster sum of squares.

**Algorithm**:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Code Example**:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        """Fit K-means clustering to data"""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.k)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Example usage
# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=1.5)

# Fit K-means
kmeans = KMeansClustering(k=4)
kmeans.fit(X)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Clusters')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clusters')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### **2. Hierarchical Clustering**

**Concept**: Build a hierarchy of clusters using agglomerative or divisive approaches.

**Code Example**:
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

class HierarchicalClustering:
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.labels = None
        
    def fit(self, X, n_clusters=None):
        """Fit hierarchical clustering to data"""
        # Compute linkage matrix
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        if n_clusters is not None:
            self.labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            
        return self
    
    def plot_dendrogram(self, figsize=(10, 6)):
        """Plot dendrogram"""
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# Example usage
hierarchical = HierarchicalClustering(method='ward')
hierarchical.fit(X, n_clusters=4)

# Plot dendrogram
hierarchical.plot_dendrogram()

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=hierarchical.labels, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.colorbar()
plt.show()
```

### **3. DBSCAN Clustering**

**Concept**: Density-based clustering that groups points in dense regions.

**Code Example**:
```python
from sklearn.cluster import DBSCAN

class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_samples = None
        
    def fit(self, X):
        """Fit DBSCAN clustering to data"""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = dbscan.fit_predict(X)
        self.core_samples = dbscan.core_sample_indices_
        return self
    
    def get_cluster_info(self):
        """Get information about clusters"""
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_core_samples': len(self.core_samples)
        }

# Example usage
dbscan = DBSCANClustering(eps=0.5, min_samples=5)
dbscan.fit(X)

# Get cluster information
info = dbscan.get_cluster_info()
print(f"Number of clusters: {info['n_clusters']}")
print(f"Number of noise points: {info['n_noise']}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.colorbar()
plt.show()
```

---

## 🧮 **Mathematical Foundations**

### **K-Means Objective Function**

**Formula**: `J = Σ(i=1 to k) Σ(x in C_i) ||x - μ_i||²`

Where:
- `k` is the number of clusters
- `C_i` is the i-th cluster
- `μ_i` is the centroid of cluster i
- `||x - μ_i||²` is the squared Euclidean distance

### **Silhouette Score**

**Formula**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`

Where:
- `a(i)` is the average distance from point i to other points in the same cluster
- `b(i)` is the minimum average distance from point i to points in other clusters

**Code Example**:
```python
from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    """Evaluate clustering performance"""
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    return silhouette_avg

# Example usage
kmeans_labels = kmeans.labels
silhouette_score_kmeans = evaluate_clustering(X, kmeans_labels)

hierarchical_labels = hierarchical.labels
silhouette_score_hierarchical = evaluate_clustering(X, hierarchical_labels)

dbscan_labels = dbscan.labels
if len(set(dbscan_labels)) > 1:  # Check if we have more than one cluster
    silhouette_score_dbscan = evaluate_clustering(X, dbscan_labels)
```

---

## 🎯 **Applications**

### **1. Customer Segmentation**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def customer_segmentation(customer_data):
    """Segment customers based on their behavior"""
    # Prepare data
    features = ['age', 'income', 'spending_score']
    X = customer_data[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeansClustering(k=5)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to data
    customer_data['cluster'] = kmeans.labels
    
    # Analyze clusters
    cluster_analysis = customer_data.groupby('cluster')[features].mean()
    print("Cluster Analysis:")
    print(cluster_analysis)
    
    return customer_data, kmeans

# Example usage
# Generate sample customer data
np.random.seed(42)
n_customers = 1000
customer_data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_customers),
    'income': np.random.normal(50000, 15000, n_customers),
    'spending_score': np.random.normal(50, 20, n_customers)
})

segmented_data, kmeans_model = customer_segmentation(customer_data)
```

### **2. Image Segmentation**

```python
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

def image_segmentation(image_path, n_colors=8):
    """Segment image into dominant colors"""
    # Load image
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    # Reshape image to list of pixels
    pixels = np.array(image).reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers (dominant colors)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Create segmented image
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_pixels.reshape(image.size[1], image.size[0], 3)
    
    return segmented_image, dominant_colors

# Example usage
# Note: You would need an actual image file
# segmented_image, colors = image_segmentation('image.jpg', n_colors=8)
```

### **3. Anomaly Detection**

```python
def anomaly_detection(X, contamination=0.1):
    """Detect anomalies using clustering"""
    from sklearn.cluster import DBSCAN
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    # Anomalies are points labeled as -1
    anomalies = X[labels == -1]
    
    print(f"Number of anomalies detected: {len(anomalies)}")
    
    return anomalies, labels

# Example usage
# Generate data with some anomalies
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomaly_data = np.random.normal(5, 0.5, (50, 2))
X_with_anomalies = np.vstack([normal_data, anomaly_data])

anomalies, labels = anomaly_detection(X_with_anomalies)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_with_anomalies[:, 0], X_with_anomalies[:, 1], 
           c=labels, cmap='viridis', alpha=0.6)
plt.scatter(anomalies[:, 0], anomalies[:, 1], 
           c='red', marker='x', s=100, label='Anomalies')
plt.title('Anomaly Detection using DBSCAN')
plt.legend()
plt.show()
```

---

## 🎯 **Interview Questions**

### **1. What is clustering and when would you use it?**

**Answer:**
Clustering is an unsupervised learning technique that groups similar data points together:
- **Use Cases**: Customer segmentation, image segmentation, anomaly detection
- **When to Use**: When you don't have labeled data but want to discover patterns
- **Benefits**: Data exploration, pattern discovery, dimensionality reduction

### **2. What's the difference between K-means and hierarchical clustering?**

**Answer:**
- **K-means**: Partitioning method, requires specifying k, faster for large datasets
- **Hierarchical**: Builds tree of clusters, doesn't require k, slower but more interpretable
- **K-means**: Good for spherical clusters, sensitive to initialization
- **Hierarchical**: Good for any cluster shape, deterministic results

### **3. How do you choose the number of clusters in K-means?**

**Answer:**
- **Elbow Method**: Plot within-cluster sum of squares vs k
- **Silhouette Analysis**: Measure how similar objects are to their own cluster
- **Gap Statistic**: Compare within-cluster dispersion to random data
- **Domain Knowledge**: Use business understanding to determine k

### **4. What are the advantages and disadvantages of DBSCAN?**

**Answer:**
**Advantages:**
- No need to specify number of clusters
- Can find clusters of arbitrary shape
- Robust to outliers
- Can identify noise points

**Disadvantages:**
- Sensitive to parameters (eps, min_samples)
- Struggles with clusters of varying densities
- Can be slow for large datasets
- Memory intensive for high-dimensional data

### **5. How do you evaluate clustering performance?**

**Answer:**
- **Internal Metrics**: Silhouette score, Calinski-Harabasz index
- **External Metrics**: Adjusted Rand index, Normalized Mutual Information
- **Visualization**: Plot clusters, dendrograms
- **Domain Validation**: Check if clusters make business sense

---

**🎉 Clustering is a powerful tool for data discovery and understanding patterns in unlabeled data!**
