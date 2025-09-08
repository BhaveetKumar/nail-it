# 🎯 Logistic Regression

> **Master logistic regression for classification: from mathematical foundations to production systems**

## 🎯 **Learning Objectives**

- Understand logistic regression theory and sigmoid function
- Implement logistic regression from scratch in Python and Go
- Master gradient descent for classification
- Handle multiclass classification with softmax
- Build production-ready classification systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Multiclass Classification](#multiclass-classification)
4. [Regularization Techniques](#regularization-techniques)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **Logistic Regression Model**

#### **Concept**
Logistic regression models the probability of binary outcomes using the logistic (sigmoid) function.

#### **Math Behind**
- **Sigmoid Function**: `σ(z) = 1/(1 + e^(-z))`
- **Logistic Model**: `P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)`
- **Log-Odds**: `log(P/(1-P)) = β₀ + β₁x₁ + ... + βₙxₙ`
- **Cost Function**: `J(β) = -1/m ∑[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, weights, bias):
        """Compute logistic regression cost (cross-entropy)"""
        m = len(y)
        z = X @ weights + bias
        predictions = self._sigmoid(z)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost
    
    def _compute_gradients(self, X, y, weights, bias):
        """Compute gradients for logistic regression"""
        m = len(y)
        z = X @ weights + bias
        predictions = self._sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * X.T @ (predictions - y)
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """Fit logistic regression model using gradient descent"""
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if np.linalg.norm(dw) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        z = X @ self.weights + self.bias
        probabilities = self._sigmoid(z)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_coefficients(self):
        """Get model coefficients"""
        return {
            'weights': self.weights,
            'bias': self.bias
        }

# Example usage
# Generate sample data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm = (X_test - X_train_mean) / X_train_std

# Train model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X_train_norm, y_train)

# Make predictions
y_pred = model.predict(X_test_norm)
y_proba = model.predict_proba(X_test_norm)

# Evaluate model
accuracy = model.score(X_test_norm, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(model.cost_history)
plt.title('Cost History - Logistic Regression')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

---

## 🎯 **Multiclass Classification**

### **Softmax Regression**

#### **Concept**
Extend logistic regression to handle multiple classes using the softmax function.

#### **Math Behind**
- **Softmax Function**: `σ(z)ᵢ = e^(zᵢ) / ∑ⱼ e^(zⱼ)`
- **Multiclass Cost**: `J(θ) = -1/m ∑ᵢ ∑ⱼ yᵢⱼ log(ŷᵢⱼ)`
- **One-Hot Encoding**: Convert class labels to binary vectors

#### **Code Example**

```python
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.cost_history = []
    
    def _softmax(self, z):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """Convert labels to one-hot encoding"""
        n_classes = len(np.unique(y))
        y_encoded = np.zeros((len(y), n_classes))
        y_encoded[np.arange(len(y)), y] = 1
        return y_encoded
    
    def _compute_cost(self, X, y_encoded, weights, bias):
        """Compute softmax regression cost"""
        m = len(y_encoded)
        z = X @ weights + bias
        predictions = self._softmax(z)
        
        # Avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y_encoded * np.log(predictions))
        return cost
    
    def _compute_gradients(self, X, y_encoded, weights, bias):
        """Compute gradients for softmax regression"""
        m = len(y_encoded)
        z = X @ weights + bias
        predictions = self._softmax(z)
        
        # Compute gradients
        dw = (1/m) * X.T @ (predictions - y_encoded)
        db = (1/m) * np.sum(predictions - y_encoded, axis=0)
        
        return dw, db
    
    def fit(self, X, y):
        """Fit softmax regression model"""
        m, n = X.shape
        self.n_classes = len(np.unique(y))
        
        # One-hot encode labels
        y_encoded = self._one_hot_encode(y)
        
        # Initialize weights and bias
        self.weights = np.random.randn(n, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X, y_encoded, self.weights, self.bias)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y_encoded, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if np.linalg.norm(dw) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        z = X @ self.weights + self.bias
        probabilities = self._softmax(z)
        return probabilities
    
    def predict(self, X):
        """Make class predictions"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
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
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm = (X_test - X_train_mean) / X_train_std

# Train softmax regression
softmax_model = SoftmaxRegression(learning_rate=0.01, max_iterations=1000)
softmax_model.fit(X_train_norm, y_train)

# Make predictions
y_pred = softmax_model.predict(X_test_norm)
y_proba = softmax_model.predict_proba(X_test_norm)

# Evaluate model
accuracy = softmax_model.score(X_test_norm, y_test)
print(f"Softmax Regression Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## 🛡️ **Regularization Techniques**

### **Regularized Logistic Regression**

#### **Concept**
Add regularization to prevent overfitting in logistic regression.

#### **Code Example**

```python
class RegularizedLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 regularization='l2', alpha=1.0, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_regularization_penalty(self, weights):
        """Compute regularization penalty"""
        if self.regularization == 'l1':
            return self.alpha * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return self.alpha * np.sum(weights ** 2)
        elif self.regularization == 'elastic_net':
            l1_penalty = self.alpha * 0.5 * np.sum(np.abs(weights))
            l2_penalty = self.alpha * 0.5 * np.sum(weights ** 2)
            return l1_penalty + l2_penalty
        else:
            return 0
    
    def _compute_regularization_gradient(self, weights):
        """Compute regularization gradient"""
        if self.regularization == 'l1':
            return self.alpha * np.sign(weights)
        elif self.regularization == 'l2':
            return 2 * self.alpha * weights
        elif self.regularization == 'elastic_net':
            l1_grad = self.alpha * 0.5 * np.sign(weights)
            l2_grad = self.alpha * 0.5 * 2 * weights
            return l1_grad + l2_grad
        else:
            return 0
    
    def _compute_cost(self, X, y, weights, bias):
        """Compute regularized logistic regression cost"""
        m = len(y)
        z = X @ weights + bias
        predictions = self._sigmoid(z)
        
        # Avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Cross-entropy cost
        cross_entropy = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Regularization penalty
        reg_penalty = self._compute_regularization_penalty(weights)
        
        return cross_entropy + reg_penalty
    
    def _compute_gradients(self, X, y, weights, bias):
        """Compute gradients with regularization"""
        m = len(y)
        z = X @ weights + bias
        predictions = self._sigmoid(z)
        
        # Cross-entropy gradients
        dw_ce = (1/m) * X.T @ (predictions - y)
        db_ce = (1/m) * np.sum(predictions - y)
        
        # Regularization gradients
        dw_reg = self._compute_regularization_gradient(weights)
        
        # Total gradients
        dw = dw_ce + dw_reg
        db = db_ce
        
        return dw, db
    
    def fit(self, X, y):
        """Fit regularized logistic regression model"""
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if np.linalg.norm(dw) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        z = X @ self.weights + self.bias
        probabilities = self._sigmoid(z)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Example usage
# Test different regularization methods
l1_model = RegularizedLogisticRegression(regularization='l1', alpha=1.0)
l2_model = RegularizedLogisticRegression(regularization='l2', alpha=1.0)
elastic_net_model = RegularizedLogisticRegression(regularization='elastic_net', alpha=1.0)

# Fit models
l1_model.fit(X_train_norm, y_train)
l2_model.fit(X_train_norm, y_train)
elastic_net_model.fit(X_train_norm, y_train)

# Evaluate models
l1_score = l1_model.score(X_test_norm, y_test)
l2_score = l2_model.score(X_test_norm, y_test)
elastic_net_score = elastic_net_model.score(X_test_norm, y_test)

print(f"L1 Regularization Accuracy: {l1_score:.4f}")
print(f"L2 Regularization Accuracy: {l2_score:.4f}")
print(f"Elastic Net Accuracy: {elastic_net_score:.4f}")

# Compare coefficients
print("\nCoefficient Comparison:")
print(f"L1 weights: {l1_model.weights}")
print(f"L2 weights: {l2_model.weights}")
print(f"Elastic Net weights: {elastic_net_model.weights}")
```

---

## 🏭 **Production Implementation**

### **Scalable Logistic Regression System**

#### **Concept**
Production systems require scalability, monitoring, and reliability for classification tasks.

#### **Code Example**

```python
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

@dataclass
class ClassificationMetrics:
    """Classification performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    training_time: float
    prediction_time: float

class ProductionLogisticRegression:
    """Production-ready logistic regression implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _preprocess_data(self, X, fit_scaler=True):
        """Preprocess data with scaling"""
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted before transforming new data")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit(self, X, y):
        """Fit the model with production features"""
        start_time = time.time()
        
        # Validate data
        if len(np.unique(y)) != 2:
            raise ValueError("Binary classification requires exactly 2 classes")
        
        # Preprocess data
        X_scaled = self._preprocess_data(X, fit_scaler=True)
        
        # Train model
        self.model = LogisticRegression(
            learning_rate=self.config.get('learning_rate', 0.01),
            max_iterations=self.config.get('max_iterations', 1000),
            regularization=self.config.get('regularization', 'l2'),
            alpha=self.config.get('alpha', 1.0)
        )
        self.model.fit(X_scaled, y)
        
        training_time = time.time() - start_time
        
        # Compute metrics
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        self.metrics = self._compute_metrics(y, predictions, probabilities, training_time)
        
        self.logger.info(f"Model trained successfully in {training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """Make predictions"""
        start_time = time.time()
        
        # Preprocess data
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        prediction_time = time.time() - start_time
        
        if self.metrics:
            self.metrics.prediction_time = prediction_time
        
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self._preprocess_data(X, fit_scaler=False)
        return self.model.predict_proba(X_scaled)
    
    def _compute_metrics(self, y_true, y_pred, y_proba, training_time):
        """Compute comprehensive classification metrics"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # AUC metrics
        auc_roc = roc_auc_score(y_true, y_proba)
        
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = np.trapz(precision_vals, recall_vals)
        
        return ClassificationMetrics(
            accuracy=accuracy, precision=precision, recall=recall,
            f1_score=f1, auc_roc=auc_roc, auc_pr=auc_pr,
            training_time=training_time, prediction_time=0
        )
    
    def plot_roc_curve(self, X, y):
        """Plot ROC curve"""
        y_proba = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.metrics.auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
    def plot_precision_recall_curve(self, X, y):
        """Plot Precision-Recall curve"""
        y_proba = self.predict_proba(X)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {self.metrics.auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self):
        """Get feature importance (coefficients)"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        coefficients = self.model.weights
        if self.feature_names:
            return dict(zip(self.feature_names, coefficients))
        return coefficients
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'model_type': 'Logistic Regression',
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance(),
            'n_features': len(self.model.weights)
        }
        
        return summary

# Example usage
config = {
    'learning_rate': 0.01,
    'max_iterations': 1000,
    'regularization': 'l2',
    'alpha': 1.0
}

# Create production model
prod_model = ProductionLogisticRegression(config)

# Train model
prod_model.fit(X_train, y_train)

# Make predictions
predictions = prod_model.predict(X_test)
probabilities = prod_model.predict_proba(X_test)

# Get model summary
summary = prod_model.get_model_summary()
print("Model Summary:", summary)

# Plot performance curves
prod_model.plot_roc_curve(X_test, y_test)
prod_model.plot_precision_recall_curve(X_test, y_test)

# Save model
prod_model.save_model('logistic_regression_model.pkl')
```

---

## 🎯 **Interview Questions**

### **Logistic Regression Theory**

#### **Q1: What is the difference between linear and logistic regression?**
**Answer**: 
- **Linear Regression**: Predicts continuous values, uses linear function, minimizes MSE
- **Logistic Regression**: Predicts probabilities, uses sigmoid function, minimizes cross-entropy
- **Output**: Linear regression outputs any real number, logistic regression outputs probabilities [0,1]

#### **Q2: Why do we use the sigmoid function in logistic regression?**
**Answer**: 
- **Bounded Output**: Sigmoid maps any real number to [0,1] range, suitable for probabilities
- **Smooth Gradient**: Differentiable everywhere, enabling gradient descent
- **Interpretable**: Output can be interpreted as probability of positive class
- **Monotonic**: Preserves order relationships in the data

#### **Q3: What is the cost function for logistic regression?**
**Answer**: 
- **Cross-Entropy Loss**: `J(θ) = -1/m ∑[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]`
- **Why Cross-Entropy**: Penalizes confident wrong predictions heavily
- **Convex**: Guarantees global minimum, unlike MSE which can have local minima
- **Probabilistic**: Directly optimizes for probability estimation

#### **Q4: How do you handle class imbalance in logistic regression?**
**Answer**: 
- **Class Weights**: Assign higher weights to minority class
- **Resampling**: SMOTE, undersampling, or oversampling
- **Threshold Tuning**: Adjust decision threshold based on business requirements
- **Cost-Sensitive Learning**: Modify cost function to penalize minority class errors more

#### **Q5: What is the difference between L1 and L2 regularization in logistic regression?**
**Answer**: 
- **L1 (Lasso)**: Adds penalty proportional to sum of absolute coefficients, can eliminate features
- **L2 (Ridge)**: Adds penalty proportional to sum of squared coefficients, shrinks coefficients
- **Feature Selection**: L1 performs automatic feature selection, L2 keeps all features
- **Sparsity**: L1 creates sparse models, L2 creates dense models

### **Implementation Questions**

#### **Q6: Implement logistic regression from scratch**
**Answer**: See the implementation above with sigmoid function, gradient descent, and cost computation.

#### **Q7: How would you handle multiclass classification with logistic regression?**
**Answer**: 
- **One-vs-Rest**: Train separate binary classifiers for each class
- **One-vs-One**: Train classifiers for each pair of classes
- **Softmax Regression**: Extend logistic regression using softmax function
- **Multinomial Logistic Regression**: Direct multiclass extension

#### **Q8: How do you evaluate logistic regression performance?**
**Answer**: 
- **Accuracy**: Overall correctness
- **Precision/Recall**: For imbalanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Confusion Matrix**: Detailed breakdown of predictions

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and scalability
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about decision trees and ensemble methods
5. **Interview**: Practice logistic regression interview questions

---

**Ready to learn about tree-based algorithms? Let's move to Decision Trees!** 🎯
