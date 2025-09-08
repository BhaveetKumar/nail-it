# 🎯 Support Vector Machines (SVM)

> **Master SVM: from mathematical foundations to kernel methods and production implementation**

## 🎯 **Learning Objectives**

- Understand SVM theory and margin maximization
- Implement SVM from scratch in Python and Go
- Master kernel methods and feature transformations
- Handle multiclass classification with SVM
- Build production-ready SVM systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Kernel Methods](#kernel-methods)
4. [Multiclass SVM](#multiclass-svm)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **SVM Theory**

#### **Concept**
SVM finds the optimal hyperplane that maximizes the margin between classes while minimizing classification errors.

#### **Math Behind**
- **Margin**: Distance between hyperplane and nearest data points
- **Support Vectors**: Data points closest to the decision boundary
- **Optimization**: Minimize `||w||²` subject to `yᵢ(wᵀxᵢ + b) ≥ 1`
- **Dual Problem**: Maximize `∑αᵢ - ½∑∑αᵢαⱼyᵢyⱼxᵢᵀxⱼ`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import cvxopt
import cvxopt.solvers

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
    
    def _linear_kernel(self, X1, X2):
        """Linear kernel: K(x, y) = x^T * y"""
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1, X2):
        """Polynomial kernel: K(x, y) = (gamma * x^T * y + coef0)^degree"""
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)"""
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X1.shape[1] * X1.var())
        
        # Compute pairwise distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        return np.exp(-self.gamma * distances)
    
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _solve_quadratic_programming(self, X, y):
        """Solve the SVM quadratic programming problem"""
        n_samples, n_features = X.shape
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        
        # Quadratic programming problem:
        # Minimize: 1/2 * alpha^T * P * alpha - q^T * alpha
        # Subject to: G * alpha <= h, A * alpha = b
        
        # P matrix (quadratic term)
        P = cvxopt.matrix(np.outer(y, y) * K)
        
        # q vector (linear term)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # G matrix (inequality constraints)
        G = cvxopt.matrix(np.vstack([
            -np.eye(n_samples),  # alpha >= 0
            np.eye(n_samples)    # alpha <= C
        ]))
        
        # h vector (inequality constraints)
        h = cvxopt.matrix(np.hstack([
            np.zeros(n_samples),  # alpha >= 0
            np.full(n_samples, self.C)  # alpha <= C
        ]))
        
        # A matrix (equality constraints)
        A = cvxopt.matrix(y.astype(float), (1, n_samples))
        
        # b vector (equality constraints)
        b = cvxopt.matrix(0.0)
        
        # Solve
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Extract solution
        alpha = np.ravel(solution['x'])
        
        # Find support vectors
        support_vector_indices = alpha > 1e-5
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y[support_vector_indices]
        self.dual_coef_ = alpha[support_vector_indices] * y[support_vector_indices]
        self.n_support_ = np.sum(support_vector_indices)
        
        # Calculate intercept
        self._calculate_intercept(X, y, alpha)
        
        return alpha
    
    def _calculate_intercept(self, X, y, alpha):
        """Calculate the intercept term"""
        # Find support vectors on the margin
        margin_support_vectors = (alpha > 1e-5) & (alpha < self.C - 1e-5)
        
        if np.any(margin_support_vectors):
            # Use support vectors on the margin
            margin_indices = np.where(margin_support_vectors)[0]
            margin_X = X[margin_indices]
            margin_y = y[margin_indices]
            margin_alpha = alpha[margin_indices]
            
            # Calculate intercept
            K_margin = self._compute_kernel(margin_X, X)
            decision_values = np.dot(K_margin, alpha * y)
            self.intercept_ = np.mean(margin_y - decision_values)
        else:
            # Fallback: use all support vectors
            if self.support_vectors_ is not None:
                K_support = self._compute_kernel(self.support_vectors_, X)
                decision_values = np.dot(K_support, alpha * y)
                self.intercept_ = np.mean(y - decision_values)
            else:
                self.intercept_ = 0.0
    
    def fit(self, X, y):
        """Fit the SVM model"""
        # Ensure binary classification
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM requires binary classification")
        
        # Convert labels to -1 and 1
        label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        y_binary = np.array([label_map[label] for label in y])
        
        # Solve quadratic programming problem
        alpha = self._solve_quadratic_programming(X, y_binary)
        
        return self
    
    def _decision_function(self, X):
        """Calculate decision function values"""
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        K = self._compute_kernel(X, self.support_vectors_)
        decision_values = np.dot(K, self.dual_coef_) + self.intercept_
        
        return decision_values
    
    def predict(self, X):
        """Make predictions"""
        decision_values = self._decision_function(X)
        predictions = np.sign(decision_values)
        
        # Convert back to original labels
        if self.support_vector_labels_ is not None:
            unique_labels = np.unique(self.support_vector_labels_)
            label_map = {-1: unique_labels[0], 1: unique_labels[1]}
            predictions = np.array([label_map[pred] for pred in predictions])
        
        return predictions
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_support_vectors(self):
        """Get support vectors"""
        return self.support_vectors_, self.support_vector_labels_

# Example usage
# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_model = SVM(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)
accuracy = svm_model.score(X_test_scaled, y_test)

print(f"SVM Accuracy: {accuracy:.4f}")
print(f"Number of support vectors: {svm_model.n_support_}")

# Get support vectors
support_vectors, support_labels = svm_model.get_support_vectors()
print(f"Support vectors shape: {support_vectors.shape}")
```

---

## 🔄 **Kernel Methods**

### **Advanced Kernel Implementations**

#### **Concept**
Kernels allow SVM to work in high-dimensional feature spaces without explicitly computing the transformation.

#### **Code Example**

```python
class AdvancedSVM(SVM):
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0):
        super().__init__(kernel, C, gamma, degree, coef0)
        self.kernel_params_ = None
    
    def _sigmoid_kernel(self, X1, X2):
        """Sigmoid kernel: K(x, y) = tanh(gamma * x^T * y + coef0)"""
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X1.shape[1] * X1.var())
        
        return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
    
    def _laplacian_kernel(self, X1, X2):
        """Laplacian kernel: K(x, y) = exp(-gamma * ||x - y||_1)"""
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X1.shape[1] * X1.var())
        
        # Compute L1 distances
        distances = np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
        
        return np.exp(-self.gamma * distances)
    
    def _chi2_kernel(self, X1, X2):
        """Chi-squared kernel: K(x, y) = exp(-gamma * sum((x-y)^2/(x+y)))"""
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X1.shape[1] * X1.var())
        
        # Compute chi-squared distances
        distances = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                diff = X1[i] - X2[j]
                sum_vals = X1[i] + X2[j]
                # Avoid division by zero
                sum_vals[sum_vals == 0] = 1e-10
                distances[i, j] = np.sum(diff**2 / sum_vals)
        
        return np.exp(-self.gamma * distances)
    
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix with advanced kernels"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'sigmoid':
            return self._sigmoid_kernel(X1, X2)
        elif self.kernel == 'laplacian':
            return self._laplacian_kernel(X1, X2)
        elif self.kernel == 'chi2':
            return self._chi2_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _grid_search_kernel_params(self, X, y, param_grid, cv=5):
        """Grid search for kernel parameters"""
        from sklearn.model_selection import GridSearchCV
        
        # Create parameter grid for different kernels
        kernel_params = {}
        
        for kernel_name, params in param_grid.items():
            kernel_params[kernel_name] = params
        
        best_score = -np.inf
        best_params = None
        best_kernel = None
        
        for kernel_name, params in kernel_params.items():
            self.kernel = kernel_name
            
            # Set default parameters
            if 'gamma' in params:
                self.gamma = params['gamma']
            if 'degree' in params:
                self.degree = params['degree']
            if 'coef0' in params:
                self.coef0 = params['coef0']
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in self._k_fold_split(X, y, cv):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Fit model
                self.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                score = self.score(X_val_fold, y_val_fold)
                scores.append(score)
            
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_kernel = kernel_name
        
        return best_kernel, best_params, best_score
    
    def _k_fold_split(self, X, y, k):
        """K-fold cross-validation split"""
        n_samples = len(X)
        fold_size = n_samples // k
        indices = np.random.permutation(n_samples)
        
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else n_samples
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, val_indices

# Example usage
# Test different kernels
X, y = make_circles(n_samples=100, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
results = {}

for kernel in kernels:
    svm = AdvancedSVM(kernel=kernel, C=1.0)
    svm.fit(X_train_scaled, y_train)
    accuracy = svm.score(X_test_scaled, y_test)
    results[kernel] = accuracy
    print(f"{kernel} kernel accuracy: {accuracy:.4f}")

# Find best kernel
best_kernel = max(results, key=results.get)
print(f"\nBest kernel: {best_kernel} with accuracy: {results[best_kernel]:.4f}")
```

---

## 🎯 **Multiclass SVM**

### **One-vs-Rest and One-vs-One**

#### **Concept**
Extend binary SVM to multiclass classification using ensemble methods.

#### **Code Example**

```python
class MulticlassSVM:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, coef0=0.0, 
                 multiclass_strategy='ovr'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.multiclass_strategy = multiclass_strategy
        self.classes_ = None
        self.binary_classifiers_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """Fit multiclass SVM"""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.multiclass_strategy == 'ovr':
            self._fit_one_vs_rest(X, y)
        elif self.multiclass_strategy == 'ovo':
            self._fit_one_vs_one(X, y)
        else:
            raise ValueError(f"Unknown multiclass strategy: {self.multiclass_strategy}")
        
        return self
    
    def _fit_one_vs_rest(self, X, y):
        """Fit one-vs-rest classifiers"""
        self.binary_classifiers_ = []
        
        for i, class_label in enumerate(self.classes_):
            # Create binary labels
            y_binary = (y == class_label).astype(int)
            
            # Train binary SVM
            svm = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma, 
                     degree=self.degree, coef0=self.coef0)
            svm.fit(X, y_binary)
            
            self.binary_classifiers_.append(svm)
    
    def _fit_one_vs_one(self, X, y):
        """Fit one-vs-one classifiers"""
        self.binary_classifiers_ = {}
        
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                class_i = self.classes_[i]
                class_j = self.classes_[j]
                
                # Get samples for these two classes
                mask = (y == class_i) | (y == class_j)
                X_pair = X[mask]
                y_pair = y[mask]
                
                # Create binary labels
                y_binary = (y_pair == class_i).astype(int)
                
                # Train binary SVM
                svm = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma, 
                         degree=self.degree, coef0=self.coef0)
                svm.fit(X_pair, y_binary)
                
                self.binary_classifiers_[(i, j)] = svm
    
    def predict(self, X):
        """Make multiclass predictions"""
        if self.multiclass_strategy == 'ovr':
            return self._predict_one_vs_rest(X)
        elif self.multiclass_strategy == 'ovo':
            return self._predict_one_vs_one(X)
    
    def _predict_one_vs_rest(self, X):
        """Predict using one-vs-rest"""
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, self.n_classes_))
        
        for i, svm in enumerate(self.binary_classifiers_):
            # Get decision function values
            decision_values = svm._decision_function(X)
            decision_scores[:, i] = decision_values
        
        # Predict class with highest decision score
        predictions = self.classes_[np.argmax(decision_scores, axis=1)]
        
        return predictions
    
    def _predict_one_vs_one(self, X):
        """Predict using one-vs-one"""
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))
        
        for (i, j), svm in self.binary_classifiers_.items():
            # Get predictions
            predictions = svm.predict(X)
            
            # Count votes
            for sample_idx in range(n_samples):
                if predictions[sample_idx] == 1:  # Class i wins
                    votes[sample_idx, i] += 1
                else:  # Class j wins
                    votes[sample_idx, j] += 1
        
        # Predict class with most votes
        predictions = self.classes_[np.argmax(votes, axis=1)]
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.multiclass_strategy == 'ovr':
            return self._predict_proba_one_vs_rest(X)
        elif self.multiclass_strategy == 'ovo':
            return self._predict_proba_one_vs_one(X)
    
    def _predict_proba_one_vs_rest(self, X):
        """Predict probabilities using one-vs-rest"""
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, self.n_classes_))
        
        for i, svm in enumerate(self.binary_classifiers_):
            decision_values = svm._decision_function(X)
            decision_scores[:, i] = decision_values
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities
    
    def _predict_proba_one_vs_one(self, X):
        """Predict probabilities using one-vs-one"""
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))
        
        for (i, j), svm in self.binary_classifiers_.items():
            predictions = svm.predict(X)
            
            for sample_idx in range(n_samples):
                if predictions[sample_idx] == 1:
                    votes[sample_idx, i] += 1
                else:
                    votes[sample_idx, j] += 1
        
        # Normalize votes to get probabilities
        total_votes = np.sum(votes, axis=1, keepdims=True)
        probabilities = votes / (total_votes + 1e-10)
        
        return probabilities
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Example usage
from sklearn.datasets import make_classification

# Generate multiclass data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test one-vs-rest
svm_ovr = MulticlassSVM(kernel='rbf', C=1.0, multiclass_strategy='ovr')
svm_ovr.fit(X_train_scaled, y_train)

y_pred_ovr = svm_ovr.predict(X_test_scaled)
y_proba_ovr = svm_ovr.predict_proba(X_test_scaled)
accuracy_ovr = svm_ovr.score(X_test_scaled, y_test)

print(f"One-vs-Rest SVM Accuracy: {accuracy_ovr:.4f}")

# Test one-vs-one
svm_ovo = MulticlassSVM(kernel='rbf', C=1.0, multiclass_strategy='ovo')
svm_ovo.fit(X_train_scaled, y_train)

y_pred_ovo = svm_ovo.predict(X_test_scaled)
y_proba_ovo = svm_ovo.predict_proba(X_test_scaled)
accuracy_ovo = svm_ovo.score(X_test_scaled, y_test)

print(f"One-vs-One SVM Accuracy: {accuracy_ovo:.4f}")

# Compare results
print(f"\nComparison:")
print(f"One-vs-Rest: {accuracy_ovr:.4f}")
print(f"One-vs-One: {accuracy_ovo:.4f}")
```

---

## 🎯 **Interview Questions**

### **SVM Theory**

#### **Q1: What is the mathematical intuition behind SVM?**
**Answer**: 
- **Margin Maximization**: SVM finds the hyperplane that maximizes the margin between classes
- **Support Vectors**: Only points closest to the decision boundary matter
- **Optimization**: Minimize `||w||²` subject to `yᵢ(wᵀxᵢ + b) ≥ 1`
- **Dual Problem**: More efficient to solve in dual space using Lagrange multipliers

#### **Q2: What is the difference between hard margin and soft margin SVM?**
**Answer**: 
- **Hard Margin**: Assumes data is linearly separable, no training errors allowed
- **Soft Margin**: Allows some training errors, uses slack variables ξᵢ
- **C Parameter**: Controls trade-off between margin maximization and error minimization
- **Large C**: Hard margin (fewer errors), **Small C**: Soft margin (more errors allowed)

#### **Q3: How do kernel methods work in SVM?**
**Answer**: 
- **Kernel Trick**: Map data to higher-dimensional space without explicit transformation
- **Inner Product**: Kernels compute inner products in feature space
- **Popular Kernels**: Linear, RBF, Polynomial, Sigmoid
- **Advantage**: Can learn non-linear decision boundaries in original space

#### **Q4: What is the difference between RBF and polynomial kernels?**
**Answer**: 
- **RBF Kernel**: `K(x,y) = exp(-γ||x-y||²)`, local similarity, infinite-dimensional feature space
- **Polynomial Kernel**: `K(x,y) = (γxᵀy + r)ᵈ`, global similarity, finite-dimensional feature space
- **RBF**: Good for smooth decision boundaries, **Polynomial**: Good for polynomial relationships
- **Parameters**: RBF uses γ, Polynomial uses γ, r, d

#### **Q5: How do you choose SVM hyperparameters?**
**Answer**: 
- **C Parameter**: Use cross-validation, balance between margin and errors
- **Kernel Selection**: Try different kernels, use domain knowledge
- **RBF γ**: Large γ (small σ) = complex boundaries, Small γ (large σ) = smooth boundaries
- **Grid Search**: Systematic search over parameter space
- **Validation**: Use validation set to avoid overfitting

### **Implementation Questions**

#### **Q6: Implement SVM from scratch**
**Answer**: See the implementation above with quadratic programming solver and kernel methods.

#### **Q7: How would you handle large-scale SVM training?**
**Answer**: 
- **Sequential Minimal Optimization (SMO)**: Efficient algorithm for large datasets
- **Stochastic Gradient Descent**: Approximate solution for very large datasets
- **Online Learning**: Update model incrementally with new data
- **Parallelization**: Train multiple SVMs in parallel
- **Approximation**: Use random features or Nyström method

#### **Q8: How do you handle imbalanced datasets with SVM?**
**Answer**: 
- **Class Weights**: Assign higher weights to minority class
- **Cost-Sensitive Learning**: Modify C parameter for different classes
- **SMOTE**: Generate synthetic samples for minority class
- **Threshold Tuning**: Adjust decision threshold based on class distribution
- **Ensemble Methods**: Combine multiple SVMs with different sampling strategies

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and scalability
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about deep learning and neural networks
5. **Interview**: Practice SVM interview questions

---

**Ready to dive into deep learning? Let's move to Neural Networks!** 🎯
