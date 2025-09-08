# 📉 Dimensionality Reduction: Simplifying Complex Data

> **Complete guide to dimensionality reduction techniques and their applications**

## 🎯 **Learning Objectives**

- Master dimensionality reduction techniques
- Understand when and why to use dimensionality reduction
- Implement PCA, t-SNE, and other techniques
- Apply dimensionality reduction to real-world problems
- Evaluate dimensionality reduction performance

## 📚 **Table of Contents**

1. [Dimensionality Reduction Fundamentals](#dimensionality-reduction-fundamentals)
2. [Linear Techniques](#linear-techniques)
3. [Nonlinear Techniques](#nonlinear-techniques)
4. [Implementation Examples](#implementation-examples)
5. [Applications](#applications)
6. [Interview Questions](#interview-questions)

---

## 📉 **Dimensionality Reduction Fundamentals**

### **Concept**

Dimensionality reduction is the process of reducing the number of features (dimensions) in a dataset while preserving important information.

### **Why Dimensionality Reduction?**

1. **Curse of Dimensionality**: Performance degrades in high dimensions
2. **Computational Efficiency**: Faster training and inference
3. **Visualization**: Plot high-dimensional data in 2D/3D
4. **Noise Reduction**: Remove irrelevant features
5. **Storage**: Reduce memory requirements

### **Types of Dimensionality Reduction**

- **Linear**: PCA, LDA, Factor Analysis
- **Nonlinear**: t-SNE, UMAP, Autoencoders
- **Feature Selection**: Filter, Wrapper, Embedded methods
- **Feature Extraction**: Transform original features

---

## 📊 **Linear Techniques**

### **1. Principal Component Analysis (PCA)**

**Concept**: Find orthogonal directions of maximum variance in the data.

**Mathematical Foundation**:
- Find eigenvectors of covariance matrix
- Sort by eigenvalues (variance explained)
- Project data onto top k eigenvectors

**Code Example**:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

class PCAAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.explained_variance_ratio_ = None
        self.components_ = None
        
    def fit(self, X):
        """Fit PCA to data"""
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_
        return self
    
    def transform(self, X):
        """Transform data to principal components"""
        return self.pca.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform data"""
        return self.pca.fit_transform(X)
    
    def plot_explained_variance(self):
        """Plot explained variance ratio"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.explained_variance_ratio_) + 1), 
                self.explained_variance_ratio_, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.show()
    
    def plot_cumulative_variance(self):
        """Plot cumulative explained variance"""
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.grid(True)
        plt.show()

# Example usage
# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCAAnalysis(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Iris Dataset')

plt.subplot(1, 3, 2)
pca.plot_explained_variance()

plt.subplot(1, 3, 3)
pca.plot_cumulative_variance()

plt.tight_layout()
plt.show()
```

### **2. Linear Discriminant Analysis (LDA)**

**Concept**: Find linear combinations that maximize class separability.

**Code Example**:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDAAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.explained_variance_ratio_ = None
        
    def fit(self, X, y):
        """Fit LDA to data"""
        self.lda.fit(X, y)
        self.explained_variance_ratio_ = self.lda.explained_variance_ratio_
        return self
    
    def transform(self, X):
        """Transform data to LDA components"""
        return self.lda.transform(X)
    
    def fit_transform(self, X, y):
        """Fit and transform data"""
        return self.lda.fit_transform(X, y)

# Example usage
lda = LDAAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Unsupervised')

plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.xlabel('First LDA Component')
plt.ylabel('Second LDA Component')
plt.title('LDA: Supervised')

plt.tight_layout()
plt.show()
```

---

## 🌊 **Nonlinear Techniques**

### **1. t-SNE (t-Distributed Stochastic Neighbor Embedding)**

**Concept**: Preserve local neighborhood structure in low-dimensional space.

**Code Example**:
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

class TSNEAnalysis:
    def __init__(self, n_components=2, perplexity=30, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.tsne = TSNE(n_components=n_components, 
                        perplexity=perplexity, 
                        random_state=random_state)
        
    def fit_transform(self, X):
        """Fit and transform data using t-SNE"""
        return self.tsne.fit_transform(X)

# Example usage
# Load digits dataset
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Apply t-SNE
tsne = TSNEAnalysis(perplexity=30)
X_tsne = tsne.fit_transform(X_digits)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE: Digits Dataset')
plt.colorbar()

# Compare with PCA
pca_digits = PCAAnalysis(n_components=2)
X_pca_digits = pca_digits.fit_transform(X_digits)

plt.subplot(1, 2, 2)
plt.scatter(X_pca_digits[:, 0], X_pca_digits[:, 1], c=y_digits, cmap='tab10')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA: Digits Dataset')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### **2. UMAP (Uniform Manifold Approximation and Projection)**

**Concept**: Preserve both local and global structure of the data.

**Code Example**:
```python
# Note: UMAP requires installation: pip install umap-learn
try:
    import umap
    
    class UMAPAnalysis:
        def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
            self.n_components = n_components
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.random_state = random_state
            self.umap = umap.UMAP(n_components=n_components,
                                 n_neighbors=n_neighbors,
                                 min_dist=min_dist,
                                 random_state=random_state)
            
        def fit_transform(self, X):
            """Fit and transform data using UMAP"""
            return self.umap.fit_transform(X)
    
    # Example usage
    umap_analysis = UMAPAnalysis(n_neighbors=15, min_dist=0.1)
    X_umap = umap_analysis.fit_transform(X_digits)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_pca_digits[:, 0], X_pca_digits[:, 1], c=y_digits, cmap='tab10')
    plt.title('PCA')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10')
    plt.title('t-SNE')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits, cmap='tab10')
    plt.title('UMAP')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```

### **3. Autoencoders**

**Concept**: Neural networks that learn to compress and reconstruct data.

**Code Example**:
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(X, encoding_dim=2, epochs=100, lr=0.001):
    """Train autoencoder for dimensionality reduction"""
    input_dim = X.shape[1]
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    
    # Initialize model
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, encoded = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Get encoded representation
    with torch.no_grad():
        _, encoded = model(X_tensor)
        encoded_np = encoded.numpy()
    
    return encoded_np, losses

# Example usage
# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# Train autoencoder
encoded_data, losses = train_autoencoder(X_scaled, encoding_dim=2, epochs=100)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=y_digits, cmap='tab10')
plt.title('Autoencoder: Digits Dataset')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='tab10')
plt.title('t-SNE: Digits Dataset')
plt.colorbar()

plt.tight_layout()
plt.show()
```

---

## 🧮 **Mathematical Foundations**

### **PCA Mathematics**

**Objective**: Maximize variance of projected data

**Mathematical Formulation**:
1. Compute covariance matrix: `C = (1/n) * X^T * X`
2. Find eigenvalues and eigenvectors: `C * v = λ * v`
3. Sort by eigenvalues (descending)
4. Project data: `Y = X * V_k`

**Code Example**:
```python
def pca_manual(X, n_components=2):
    """Manual PCA implementation"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_pca = X_centered @ components
    
    return X_pca, eigenvalues, eigenvectors

# Example usage
X_pca_manual, eigenvalues, eigenvectors = pca_manual(X, n_components=2)

# Compare with sklearn
from sklearn.decomposition import PCA
pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X)

print(f"Manual PCA shape: {X_pca_manual.shape}")
print(f"Sklearn PCA shape: {X_pca_sklearn.shape}")
print(f"Explained variance ratio: {pca_sklearn.explained_variance_ratio_}")
```

---

## 🎯 **Applications**

### **1. Image Compression**

```python
def image_compression_pca(image_path, n_components=50):
    """Compress image using PCA"""
    # Load image
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(image_array)
    
    # Reconstruct image
    reconstructed = pca.inverse_transform(compressed)
    
    # Calculate compression ratio
    original_size = image_array.size
    compressed_size = compressed.size + pca.components_.size
    compression_ratio = original_size / compressed_size
    
    return reconstructed, compression_ratio, pca.explained_variance_ratio_

# Example usage
# Note: You would need an actual image file
# reconstructed, ratio, variance = image_compression_pca('image.jpg', n_components=50)
# print(f"Compression ratio: {ratio:.2f}")
# print(f"Variance explained: {variance.sum():.3f}")
```

### **2. Feature Selection**

```python
from sklearn.feature_selection import SelectKBest, f_classif

def feature_selection_analysis(X, y, k=10):
    """Analyze feature selection methods"""
    # SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected features
    selected_features = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    return X_selected, selected_features, feature_scores

# Example usage
X_selected, features, scores = feature_selection_analysis(X_digits, y_digits, k=20)
print(f"Selected features: {features}")
print(f"Feature scores: {scores[features]}")
```

### **3. Data Visualization**

```python
def visualize_high_dimensional_data(X, y, method='tsne'):
    """Visualize high-dimensional data in 2D"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    X_reduced = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'{method.upper()}: High-dimensional Data Visualization')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.show()
    
    return X_reduced

# Example usage
X_tsne_viz = visualize_high_dimensional_data(X_digits, y_digits, method='tsne')
X_pca_viz = visualize_high_dimensional_data(X_digits, y_digits, method='pca')
```

---

## 🎯 **Interview Questions**

### **1. What is dimensionality reduction and why is it important?**

**Answer:**
Dimensionality reduction is the process of reducing the number of features while preserving important information:
- **Curse of Dimensionality**: Performance degrades in high dimensions
- **Computational Efficiency**: Faster training and inference
- **Visualization**: Plot high-dimensional data in 2D/3D
- **Noise Reduction**: Remove irrelevant features
- **Storage**: Reduce memory requirements

### **2. What's the difference between PCA and LDA?**

**Answer:**
- **PCA**: Unsupervised, maximizes variance, finds directions of maximum spread
- **LDA**: Supervised, maximizes class separability, finds directions that best separate classes
- **PCA**: Good for data exploration and visualization
- **LDA**: Good for classification tasks with labeled data

### **3. When would you use t-SNE vs PCA?**

**Answer:**
**Use PCA when:**
- Linear relationships are sufficient
- Need to preserve global structure
- Want fast computation
- Need to handle new data points

**Use t-SNE when:**
- Need to preserve local neighborhood structure
- Data has complex nonlinear relationships
- Visualization is the primary goal
- Can afford slower computation

### **4. How do you choose the number of components in PCA?**

**Answer:**
- **Elbow Method**: Plot explained variance vs number of components
- **Cumulative Variance**: Choose components that explain 95% of variance
- **Scree Plot**: Look for the "elbow" in the plot
- **Cross-validation**: Use downstream task performance
- **Domain Knowledge**: Use business understanding

### **5. What are the limitations of dimensionality reduction?**

**Answer:**
- **Information Loss**: Some information is always lost
- **Interpretability**: Reduced features may be harder to interpret
- **Computational Cost**: Some methods are computationally expensive
- **Parameter Sensitivity**: Results depend on hyperparameters
- **New Data**: Some methods can't handle new data points

---

**🎉 Dimensionality reduction is essential for handling high-dimensional data and improving model performance!**
