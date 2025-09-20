# Machine Learning

## Table of Contents

1. [Overview](#overview)
2. [Supervised Learning](#supervised-learning)
3. [Unsupervised Learning](#unsupervised-learning)
4. [Deep Learning](#deep-learning)
5. [Natural Language Processing](#natural-language-processing)
6. [Computer Vision](#computer-vision)
7. [Model Deployment](#model-deployment)
8. [MLOps](#mlops)
9. [Implementations](#implementations)
10. [Follow-up Questions](#follow-up-questions)
11. [Sources](#sources)
12. [Projects](#projects)

## Overview

### Learning Objectives

- Master supervised and unsupervised learning algorithms
- Implement deep learning models with neural networks
- Apply NLP and computer vision techniques
- Deploy and manage ML models in production
- Implement MLOps practices and pipelines
- Optimize model performance and scalability

### What is Machine Learning?

Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

## Supervised Learning

### 1. Linear Regression

#### Linear Regression Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, y_true, y_pred):
        return (1/(2*len(y_true))) * np.sum((y_pred - y_true)**2)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R²: {train_score:.4f}")
print(f"Testing R²: {test_score:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, alpha=0.7, label='Actual')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Predictions')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model.cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Decision Trees

#### Decision Tree Implementation
```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(set(y)) == 1:
            return self._create_leaf(y)
        
        # Find best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return self._create_leaf(y)
        
        feature_idx, threshold = best_split
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                gini = self._calculate_gini(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _calculate_gini(self, left_y, right_y):
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0
        
        gini_left = 1 - sum((np.sum(left_y == label) / n_left) ** 2 for label in np.unique(left_y))
        gini_right = 1 - sum((np.sum(right_y == label) / n_right) ** 2 for label in np.unique(right_y))
        
        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    
    def _create_leaf(self, y):
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, self.tree))
        return np.array(predictions)
    
    def _predict_sample(self, sample, node):
        if isinstance(node, (int, float, str)):  # Leaf node
            return node
        
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
dt = DecisionTree(max_depth=3, min_samples_split=5)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
```

### 3. Random Forest

#### Random Forest Implementation
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Train multiple decision trees
        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Random feature selection
            n_features_to_select = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, size=n_features_to_select, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]
            
            # Train decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Store tree with feature indices
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
            
            # Update feature importances (simplified)
            for j, feature_idx in enumerate(feature_indices):
                self.feature_importances_[feature_idx] += 1
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators
    
    def predict(self, X):
        predictions = []
        for sample in X:
            votes = []
            for tree_info in self.trees:
                tree = tree_info['tree']
                feature_indices = tree_info['feature_indices']
                sample_subset = sample[feature_indices].reshape(1, -1)
                prediction = tree.predict(sample_subset)[0]
                votes.append(prediction)
            
            # Majority voting
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            votes = []
            for tree_info in self.trees:
                tree = tree_info['tree']
                feature_indices = tree_info['feature_indices']
                sample_subset = sample[feature_indices].reshape(1, -1)
                prediction = tree.predict(sample_subset)[0]
                votes.append(prediction)
            
            # Calculate probabilities
            vote_counts = Counter(votes)
            total_votes = len(votes)
            proba = [vote_counts.get(label, 0) / total_votes for label in sorted(set(votes))]
            probabilities.append(proba)
        
        return np.array(probabilities)

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest
rf = RandomForest(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Feature importances
print("Feature Importances:")
for i, importance in enumerate(rf.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")
```

## Unsupervised Learning

### 1. K-Means Clustering

#### K-Means Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for iteration in range(self.max_iterations):
            # Assign points to closest centroid
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        # Calculate inertia
        self.inertia = self._calculate_inertia(X)
    
    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        return distances
    
    def _calculate_inertia(self, X):
        inertia = 0
        for i, centroid in enumerate(self.centroids):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
    
    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# Example usage
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Predictions
y_pred = kmeans.predict(X)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print(f"Inertia: {kmeans.inertia:.4f}")
```

### 2. Principal Component Analysis (PCA)

#### PCA Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select number of components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.n_components = min(self.n_components, X.shape[1])
        
        # Store components and explained variance ratio
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
    
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Example usage
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Original Data (First 2 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('PCA Transformed Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
```

## Deep Learning

### 1. Neural Network Implementation

#### Neural Network from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activations = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.1
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:  # Output layer
                a = self.sigmoid(z)
            else:  # Hidden layers
                a = self.relu(z)
            self.activations.append(a)
            current_input = a
        
        return current_input
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer error
        dz = output - y
        dw = (1/m) * np.dot(self.activations[-2].T, dz)
        db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights[-1] -= self.learning_rate * dw
        self.biases[-1] -= self.learning_rate * db
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[i + 1].T)
            dz = da * self.relu_derivative(self.activations[i + 1])
            dw = (1/m) * np.dot(self.activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X, y, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute cost
            cost = self.compute_cost(y, output)
            costs.append(cost)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        return costs
    
    def compute_cost(self, y_true, y_pred):
        m = y_true.shape[0]
        cost = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return cost
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape y for neural network
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create and train neural network
nn = NeuralNetwork([2, 4, 4, 1], learning_rate=0.01)
costs = nn.train(X_train, y_train, epochs=1000)

# Make predictions
y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {accuracy:.4f}")

# Plot training progress
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.title('Training Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred.flatten(), cmap='viridis', alpha=0.7)
plt.title('Neural Network Predictions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## Natural Language Processing

### 1. Text Preprocessing

#### Text Preprocessing Pipeline
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem tokens
        tokens = self.stem_tokens(tokens)
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        return self.vectorizer.fit_transform(preprocessed_texts)
    
    def transform(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        return self.vectorizer.transform(preprocessed_texts)

# Example usage
texts = [
    "This is a great product! I love it.",
    "The service was terrible. Very disappointed.",
    "Amazing experience, would recommend to everyone.",
    "Waste of money. Don't buy this product.",
    "Excellent quality and fast delivery."
]

labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative

# Preprocess texts
preprocessor = TextPreprocessor()
X = preprocessor.fit_transform(texts)

print("Original texts:")
for i, text in enumerate(texts):
    print(f"{i+1}. {text}")

print("\nPreprocessed texts:")
for i, text in enumerate(texts):
    preprocessed = preprocessor.preprocess(text)
    print(f"{i+1}. {preprocessed}")

print(f"\nTF-IDF Matrix shape: {X.shape}")
print(f"Feature names: {preprocessor.vectorizer.get_feature_names_out()[:10]}...")
```

### 2. Sentiment Analysis

#### Sentiment Analysis Model
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = LogisticRegression(random_state=42)
    
    def fit(self, texts, labels):
        # Preprocess texts
        X = self.preprocessor.fit_transform(texts)
        
        # Train model
        self.model.fit(X, labels)
        
        return self
    
    def predict(self, texts):
        # Preprocess texts
        X = self.preprocessor.transform(texts)
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_proba(self, texts):
        # Preprocess texts
        X = self.preprocessor.transform(texts)
        
        # Get prediction probabilities
        return self.model.predict_proba(X)

# Example usage
# Sample data
texts = [
    "This movie is absolutely fantastic!",
    "I hate this product, it's terrible.",
    "The service was okay, nothing special.",
    "Amazing experience, highly recommended!",
    "Waste of time and money.",
    "Great quality and fast delivery.",
    "This is the worst thing I've ever bought.",
    "Excellent customer service and support.",
    "Poor quality, would not recommend.",
    "Outstanding product, love it!"
]

labels = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  # 1: positive, 0: negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Train sentiment analyzer
analyzer = SentimentAnalyzer()
analyzer.fit(X_train, y_train)

# Make predictions
y_pred = analyzer.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Sentiment Analysis Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test on new texts
new_texts = [
    "This is the best product ever!",
    "I'm so disappointed with this purchase.",
    "It's okay, nothing special."
]

predictions = analyzer.predict(new_texts)
probabilities = analyzer.predict_proba(new_texts)

print("\nNew Text Predictions:")
for text, pred, prob in zip(new_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = max(prob)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
    print()
```

## Model Deployment

### 1. Model Serialization

#### Model Persistence
```python
import pickle
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelManager:
    def __init__(self, model=None):
        self.model = model
        self.metadata = {}
    
    def train_model(self, X, y):
        # Train a sample model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Store metadata
        self.metadata = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': 100,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'classes': list(self.model.classes_)
        }
        
        return self.model
    
    def save_model(self, filepath):
        # Save model using joblib
        model_path = f"{filepath}_model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath):
        # Load model
        model_path = f"{filepath}_model.pkl"
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Model loaded from {model_path}")
        print(f"Metadata loaded from {metadata_path}")
        
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        return self.model.predict_proba(X)

# Example usage
# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model_manager = ModelManager()
model_manager.train_model(X_train, y_train)
model_manager.save_model("my_model")

# Load model and make predictions
loaded_manager = ModelManager()
loaded_manager.load_model("my_model")

# Test predictions
y_pred = loaded_manager.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Loaded Model Accuracy: {accuracy:.4f}")

# Print metadata
print("\nModel Metadata:")
for key, value in loaded_manager.metadata.items():
    print(f"{key}: {value}")
```

### 2. API Deployment

#### Flask API for ML Model
```python
from flask import Flask, request, jsonify
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load model and metadata
model = joblib.load('my_model_model.pkl')
with open('my_model_metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Validate input shape
        if features.shape[1] != metadata['n_features']:
            return jsonify({
                'error': f'Expected {metadata["n_features"]} features, got {features.shape[1]}'
            }), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'confidence': float(max(probabilities))
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_type': metadata['model_type']})

@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Follow-up Questions

### 1. Supervised Learning
**Q: What's the difference between bias and variance in machine learning?**
A: Bias is the error due to oversimplified assumptions, while variance is the error due to sensitivity to small fluctuations in training data.

### 2. Unsupervised Learning
**Q: How do you choose the optimal number of clusters in K-means?**
A: Use methods like the elbow method, silhouette analysis, or gap statistic to find the optimal number of clusters.

### 3. Deep Learning
**Q: What's the vanishing gradient problem and how is it solved?**
A: The vanishing gradient problem occurs when gradients become very small during backpropagation. It's solved using techniques like ReLU activation, batch normalization, and residual connections.

## Sources

### Books
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Hands-On Machine Learning** by Aurélien Géron

### Online Resources
- **Scikit-learn Documentation** - Machine learning library
- **TensorFlow Tutorials** - Deep learning framework
- **PyTorch Tutorials** - Deep learning framework

## Projects

### 1. Image Classification System
**Objective**: Build an image classification system
**Requirements**: CNN, data augmentation, model optimization
**Deliverables**: Complete image classification pipeline

### 2. Natural Language Processing Pipeline
**Objective**: Create an NLP pipeline for text analysis
**Requirements**: Text preprocessing, feature extraction, sentiment analysis
**Deliverables**: End-to-end NLP system

### 3. MLOps Platform
**Objective**: Build an MLOps platform for model deployment
**Requirements**: Model versioning, monitoring, CI/CD
**Deliverables**: Complete MLOps infrastructure

---

**Next**: [Cloud Architecture](../../../README.md) | **Previous**: [Distributed Systems](../../../README.md) | **Up**: [Phase 2](README.md/)

