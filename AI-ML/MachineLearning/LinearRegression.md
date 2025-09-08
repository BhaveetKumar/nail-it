# 📈 Linear Regression

> **Master linear regression from mathematical foundations to production implementation**

## 🎯 **Learning Objectives**

- Understand linear regression theory and assumptions
- Implement linear regression from scratch in Python and Go
- Master gradient descent optimization
- Handle overfitting with regularization
- Build production-ready linear regression systems

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations)
2. [Implementation from Scratch](#implementation-from-scratch)
3. [Gradient Descent Optimization](#gradient-descent-optimization)
4. [Regularization Techniques](#regularization-techniques)
5. [Production Implementation](#production-implementation)
6. [Interview Questions](#interview-questions)

---

## 🧮 **Mathematical Foundations**

### **Linear Regression Model**

#### **Concept**
Linear regression models the relationship between a dependent variable and one or more independent variables using a linear function.

#### **Math Behind**
- **Simple Linear Regression**: `y = β₀ + β₁x + ε`
- **Multiple Linear Regression**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`
- **Matrix Form**: `Y = Xβ + ε`
- **Normal Equation**: `β = (XᵀX)⁻¹XᵀY`

#### **Assumptions**
1. **Linearity**: Relationship between variables is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_intercept(self, X):
        """Add intercept term to features"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _compute_cost(self, X, y, weights):
        """Compute mean squared error cost"""
        predictions = X @ weights
        mse = np.mean((predictions - y) ** 2)
        return mse
    
    def _compute_gradients(self, X, y, weights):
        """Compute gradients for gradient descent"""
        predictions = X @ weights
        error = predictions - y
        gradients = (1 / len(X)) * X.T @ error
        return gradients
    
    def fit_normal_equation(self, X, y):
        """Fit using normal equation (closed-form solution)"""
        X_with_intercept = self._add_intercept(X)
        
        # Normal equation: β = (XᵀX)⁻¹XᵀY
        XTX = X_with_intercept.T @ X_with_intercept
        XTX_inv = np.linalg.inv(XTX)
        XTY = X_with_intercept.T @ y
        
        weights = XTX_inv @ XTY
        
        self.bias = weights[0]
        self.weights = weights[1:]
        
        return self
    
    def fit_gradient_descent(self, X, y):
        """Fit using gradient descent"""
        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]
        
        # Initialize weights
        weights = np.random.randn(n_features) * 0.01
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X_with_intercept, y, weights)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = self._compute_gradients(X_with_intercept, y, weights)
            
            # Update weights
            weights = weights - self.learning_rate * gradients
            
            # Check convergence
            if np.linalg.norm(gradients) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        self.bias = weights[0]
        self.weights = weights[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_coefficients(self):
        """Get model coefficients"""
        return {
            'bias': self.bias,
            'weights': self.weights
        }

# Example usage
# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_train_mean) / X_train_std
X_test_norm = (X_test - X_train_mean) / X_train_std

# Train models
model_normal = LinearRegression()
model_gd = LinearRegression(learning_rate=0.01, max_iterations=1000)

model_normal.fit_normal_equation(X_train_norm, y_train)
model_gd.fit_gradient_descent(X_train_norm, y_train)

# Evaluate models
normal_score = model_normal.score(X_test_norm, y_test)
gd_score = model_gd.score(X_test_norm, y_test)

print(f"Normal Equation R²: {normal_score:.4f}")
print(f"Gradient Descent R²: {gd_score:.4f}")

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(model_gd.cost_history)
plt.title('Cost History - Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()
```

---

## 🎯 **Gradient Descent Optimization**

### **Advanced Optimization Techniques**

#### **Concept**
Gradient descent variants improve convergence speed and stability.

#### **Code Example**

```python
class AdvancedGradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.cost_history = []
    
    def stochastic_gradient_descent(self, X, y, batch_size=32):
        """Stochastic Gradient Descent with mini-batches"""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize weights
        weights = np.random.randn(n_features + 1) * 0.01
        X_with_intercept = self._add_intercept(X)
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_intercept[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Compute gradients
                predictions = batch_X @ weights
                error = predictions - batch_y
                gradients = (1 / len(batch_X)) * batch_X.T @ error
                
                # Update weights
                weights = weights - self.learning_rate * gradients
            
            # Compute cost
            cost = self._compute_cost(X_with_intercept, y, weights)
            self.cost_history.append(cost)
        
        return weights
    
    def momentum_gradient_descent(self, X, y, momentum=0.9):
        """Gradient descent with momentum"""
        n_features = X.shape[1]
        X_with_intercept = self._add_intercept(X)
        
        # Initialize weights and velocity
        weights = np.random.randn(n_features + 1) * 0.01
        velocity = np.zeros(n_features + 1)
        
        for iteration in range(self.max_iterations):
            # Compute gradients
            gradients = self._compute_gradients(X_with_intercept, y, weights)
            
            # Update velocity
            velocity = momentum * velocity + self.learning_rate * gradients
            
            # Update weights
            weights = weights - velocity
            
            # Compute cost
            cost = self._compute_cost(X_with_intercept, y, weights)
            self.cost_history.append(cost)
        
        return weights
    
    def adaptive_gradient_descent(self, X, y, epsilon=1e-8):
        """AdaGrad - Adaptive gradient descent"""
        n_features = X.shape[1]
        X_with_intercept = self._add_intercept(X)
        
        # Initialize weights and squared gradients
        weights = np.random.randn(n_features + 1) * 0.01
        squared_gradients = np.zeros(n_features + 1)
        
        for iteration in range(self.max_iterations):
            # Compute gradients
            gradients = self._compute_gradients(X_with_intercept, y, weights)
            
            # Update squared gradients
            squared_gradients += gradients ** 2
            
            # Update weights with adaptive learning rate
            adaptive_lr = self.learning_rate / (np.sqrt(squared_gradients) + epsilon)
            weights = weights - adaptive_lr * gradients
            
            # Compute cost
            cost = self._compute_cost(X_with_intercept, y, weights)
            self.cost_history.append(cost)
        
        return weights

# Example usage
advanced_gd = AdvancedGradientDescent(learning_rate=0.01, max_iterations=1000)

# Test different optimization methods
weights_sgd = advanced_gd.stochastic_gradient_descent(X_train_norm, y_train)
weights_momentum = advanced_gd.momentum_gradient_descent(X_train_norm, y_train)
weights_adagrad = advanced_gd.adaptive_gradient_descent(X_train_norm, y_train)

print("Advanced gradient descent methods implemented successfully")
```

---

## 🛡️ **Regularization Techniques**

### **Ridge and Lasso Regression**

#### **Concept**
Regularization prevents overfitting by adding penalty terms to the cost function.

#### **Math Behind**
- **Ridge Regression**: `J(β) = MSE + α∑βᵢ²`
- **Lasso Regression**: `J(β) = MSE + α∑|βᵢ|`
- **Elastic Net**: `J(β) = MSE + α₁∑|βᵢ| + α₂∑βᵢ²`

#### **Code Example**

```python
class RegularizedLinearRegression:
    def __init__(self, alpha=1.0, regularization='ridge', learning_rate=0.01, max_iterations=1000):
        self.alpha = alpha
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _compute_regularization_penalty(self, weights):
        """Compute regularization penalty"""
        if self.regularization == 'ridge':
            return self.alpha * np.sum(weights ** 2)
        elif self.regularization == 'lasso':
            return self.alpha * np.sum(np.abs(weights))
        elif self.regularization == 'elastic_net':
            l1_penalty = self.alpha * 0.5 * np.sum(np.abs(weights))
            l2_penalty = self.alpha * 0.5 * np.sum(weights ** 2)
            return l1_penalty + l2_penalty
        else:
            return 0
    
    def _compute_regularization_gradient(self, weights):
        """Compute regularization gradient"""
        if self.regularization == 'ridge':
            return 2 * self.alpha * weights
        elif self.regularization == 'lasso':
            return self.alpha * np.sign(weights)
        elif self.regularization == 'elastic_net':
            l1_grad = self.alpha * 0.5 * np.sign(weights)
            l2_grad = self.alpha * 0.5 * 2 * weights
            return l1_grad + l2_grad
        else:
            return 0
    
    def fit(self, X, y):
        """Fit regularized linear regression"""
        X_with_intercept = self._add_intercept(X)
        n_features = X_with_intercept.shape[1]
        
        # Initialize weights
        weights = np.random.randn(n_features) * 0.01
        
        for iteration in range(self.max_iterations):
            # Compute predictions
            predictions = X_with_intercept @ weights
            
            # Compute cost
            mse = np.mean((predictions - y) ** 2)
            regularization_penalty = self._compute_regularization_penalty(weights[1:])  # Exclude bias
            total_cost = mse + regularization_penalty
            self.cost_history.append(total_cost)
            
            # Compute gradients
            mse_gradients = (1 / len(X)) * X_with_intercept.T @ (predictions - y)
            reg_gradients = np.zeros_like(weights)
            reg_gradients[1:] = self._compute_regularization_gradient(weights[1:])  # Exclude bias
            
            total_gradients = mse_gradients + reg_gradients
            
            # Update weights
            weights = weights - self.learning_rate * total_gradients
        
        self.bias = weights[0]
        self.weights = weights[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return X @ self.weights + self.bias
    
    def get_coefficients(self):
        """Get model coefficients"""
        return {
            'bias': self.bias,
            'weights': self.weights
        }

# Example usage
# Test different regularization methods
ridge_model = RegularizedLinearRegression(alpha=1.0, regularization='ridge')
lasso_model = RegularizedLinearRegression(alpha=1.0, regularization='lasso')
elastic_net_model = RegularizedLinearRegression(alpha=1.0, regularization='elastic_net')

# Fit models
ridge_model.fit(X_train_norm, y_train)
lasso_model.fit(X_train_norm, y_train)
elastic_net_model.fit(X_train_norm, y_train)

# Evaluate models
ridge_score = ridge_model.score(X_test_norm, y_test)
lasso_score = lasso_model.score(X_test_norm, y_test)
elastic_net_score = elastic_net_model.score(X_test_norm, y_test)

print(f"Ridge R²: {ridge_score:.4f}")
print(f"Lasso R²: {lasso_score:.4f}")
print(f"Elastic Net R²: {elastic_net_score:.4f}")

# Compare coefficients
print("\nCoefficient Comparison:")
print(f"Ridge weights: {ridge_model.get_coefficients()['weights']}")
print(f"Lasso weights: {lasso_model.get_coefficients()['weights']}")
print(f"Elastic Net weights: {elastic_net_model.get_coefficients()['weights']}")
```

---

## 🏭 **Production Implementation**

### **Scalable Linear Regression System**

#### **Concept**
Production systems require scalability, monitoring, and reliability.

#### **Code Example**

```python
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mse: float
    rmse: float
    r2_score: float
    mae: float
    training_time: float
    prediction_time: float

class ModelValidator:
    """Model validation and quality checks"""
    
    @staticmethod
    def validate_data(X, y):
        """Validate input data"""
        if X is None or y is None:
            raise ValueError("Input data cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if X.shape[0] == 0:
            raise ValueError("Input data cannot be empty")
        
        return True
    
    @staticmethod
    def validate_predictions(predictions, y_true):
        """Validate predictions"""
        if predictions is None:
            raise ValueError("Predictions cannot be None")
        
        if len(predictions) != len(y_true):
            raise ValueError("Predictions and true values must have the same length")
        
        return True

class ProductionLinearRegression:
    """Production-ready linear regression implementation"""
    
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
        ModelValidator.validate_data(X, y)
        
        # Preprocess data
        X_scaled = self._preprocess_data(X, fit_scaler=True)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        training_time = time.time() - start_time
        
        # Compute metrics
        predictions = self.predict(X)
        self.metrics = self._compute_metrics(y, predictions, training_time)
        
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
    
    def _compute_metrics(self, y_true, y_pred, training_time):
        """Compute model metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        return ModelMetrics(
            mse=mse, rmse=rmse, r2_score=r2, mae=mae,
            training_time=training_time, prediction_time=0
        )
    
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
        
        coefficients = self.model.coef_
        if self.feature_names:
            return dict(zip(self.feature_names, coefficients))
        return coefficients
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'model_type': 'Linear Regression',
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance(),
            'n_features': len(self.model.coef_)
        }
        
        return summary

# Example usage
config = {
    'learning_rate': 0.01,
    'max_iterations': 1000,
    'regularization': 'ridge',
    'alpha': 1.0
}

# Create production model
prod_model = ProductionLinearRegression(config)

# Train model
prod_model.fit(X_train, y_train)

# Make predictions
predictions = prod_model.predict(X_test)

# Get model summary
summary = prod_model.get_model_summary()
print("Model Summary:", summary)

# Save model
prod_model.save_model('linear_regression_model.pkl')

# Load model
loaded_model = ProductionLinearRegression(config)
loaded_model.load_model('linear_regression_model.pkl')
```

---

## 🎯 **Interview Questions**

### **Linear Regression Theory**

#### **Q1: What are the assumptions of linear regression?**
**Answer**: 
1. **Linearity**: Relationship between variables is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Independent variables are not highly correlated

#### **Q2: What is the difference between R² and adjusted R²?**
**Answer**: R² measures the proportion of variance explained by the model. Adjusted R² penalizes for the number of predictors, preventing overfitting. Adjusted R² = 1 - (1-R²)(n-1)/(n-k-1) where n is sample size and k is number of predictors.

#### **Q3: How do you handle multicollinearity in linear regression?**
**Answer**: 
- **Detection**: Calculate VIF (Variance Inflation Factor)
- **Solutions**: Remove highly correlated variables, use regularization (Ridge), or apply PCA
- **VIF > 10** indicates severe multicollinearity

#### **Q4: Explain the bias-variance tradeoff in linear regression**
**Answer**: 
- **Bias**: Error from oversimplifying assumptions
- **Variance**: Error from sensitivity to small fluctuations in training data
- **Tradeoff**: Increasing model complexity reduces bias but increases variance
- **Optimal**: Find the balance that minimizes total error

#### **Q5: What is the difference between Ridge and Lasso regularization?**
**Answer**: 
- **Ridge (L2)**: Adds penalty proportional to sum of squared coefficients, shrinks coefficients but doesn't eliminate them
- **Lasso (L1)**: Adds penalty proportional to sum of absolute coefficients, can eliminate features by setting coefficients to zero
- **Use Ridge** when you want to keep all features, **use Lasso** for feature selection

### **Implementation Questions**

#### **Q6: Implement linear regression from scratch**
**Answer**: See the implementation above with normal equation and gradient descent methods.

#### **Q7: How would you scale linear regression to handle millions of samples?**
**Answer**: 
- **Stochastic Gradient Descent**: Use mini-batches
- **Distributed Computing**: Use Spark or Dask
- **Online Learning**: Update model incrementally
- **Feature Selection**: Reduce dimensionality
- **Caching**: Cache frequently used computations

#### **Q8: How do you handle missing values in linear regression?**
**Answer**: 
- **Listwise deletion**: Remove rows with missing values
- **Imputation**: Fill missing values with mean, median, or mode
- **Advanced methods**: Use iterative imputation or model-based imputation
- **Consider**: Impact on model assumptions and performance

---

## 🚀 **Next Steps**

1. **Practice**: Implement all variants and test with different datasets
2. **Optimize**: Focus on performance and scalability
3. **Deploy**: Build production systems with monitoring
4. **Extend**: Learn about polynomial regression and feature engineering
5. **Interview**: Practice linear regression interview questions

---

**Ready to learn about classification? Let's move to Logistic Regression!** 🎯
