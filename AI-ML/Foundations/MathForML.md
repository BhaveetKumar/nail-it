# 📐 Mathematics for Machine Learning

> **Essential mathematical foundations for AI/ML: Linear Algebra, Calculus, and Optimization**

## 🎯 **Learning Objectives**

- Master linear algebra operations and their ML applications
- Understand calculus concepts for optimization and backpropagation
- Learn optimization techniques for model training
- Implement mathematical concepts in Python and Go
- Apply mathematical foundations to real ML problems

## 📚 **Table of Contents**

1. [Linear Algebra Fundamentals](#linear-algebra-fundamentals)
2. [Calculus for ML](#calculus-for-ml)
3. [Optimization Theory](#optimization-theory)
4. [Information Theory](#information-theory)
5. [Statistical Learning Theory](#statistical-learning-theory)
6. [Implementation Examples](#implementation-examples)
7. [Interview Questions](#interview-questions)

---

## 🔢 **Linear Algebra Fundamentals**

### **Vectors and Vector Operations**

#### **Concept**

Vectors are fundamental building blocks in ML, representing data points, features, and model parameters.

#### **Math Behind**

- **Vector Addition**: `v + w = [v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ]`
- **Scalar Multiplication**: `cv = [cv₁, cv₂, ..., cvₙ]`
- **Dot Product**: `v · w = Σᵢ vᵢwᵢ`
- **Vector Norm**: `||v|| = √(Σᵢ vᵢ²)`

#### **Code Example**

```python
import numpy as np
import matplotlib.pyplot as plt

class VectorOperations:
    def __init__(self):
        self.vectors = {}

    def add_vectors(self, v1, v2):
        """Add two vectors element-wise"""
        return np.array(v1) + np.array(v2)

    def dot_product(self, v1, v2):
        """Calculate dot product of two vectors"""
        return np.dot(v1, v2)

    def vector_norm(self, v):
        """Calculate L2 norm of vector"""
        return np.linalg.norm(v)

    def cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between vectors"""
        dot_product = self.dot_product(v1, v2)
        norms = self.vector_norm(v1) * self.vector_norm(v2)
        return dot_product / norms if norms != 0 else 0

# Example usage
vector_ops = VectorOperations()

# Create sample vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector addition: {vector_ops.add_vectors(v1, v2)}")
print(f"Dot product: {vector_ops.dot_product(v1, v2)}")
print(f"Vector norm: {vector_ops.vector_norm(v1)}")
print(f"Cosine similarity: {vector_ops.cosine_similarity(v1, v2)}")
```

#### **Golang Implementation**

```go
package main

import (
    "fmt"
    "math"
)

type Vector struct {
    Data []float64
}

func NewVector(data []float64) *Vector {
    return &Vector{Data: data}
}

func (v *Vector) Add(other *Vector) *Vector {
    if len(v.Data) != len(other.Data) {
        panic("Vectors must have same length")
    }

    result := make([]float64, len(v.Data))
    for i := range v.Data {
        result[i] = v.Data[i] + other.Data[i]
    }
    return NewVector(result)
}

func (v *Vector) DotProduct(other *Vector) float64 {
    if len(v.Data) != len(other.Data) {
        panic("Vectors must have same length")
    }

    result := 0.0
    for i := range v.Data {
        result += v.Data[i] * other.Data[i]
    }
    return result
}

func (v *Vector) Norm() float64 {
    sum := 0.0
    for _, val := range v.Data {
        sum += val * val
    }
    return math.Sqrt(sum)
}

func (v *Vector) CosineSimilarity(other *Vector) float64 {
    dotProduct := v.DotProduct(other)
    norms := v.Norm() * other.Norm()
    if norms == 0 {
        return 0
    }
    return dotProduct / norms
}

func main() {
    v1 := NewVector([]float64{1, 2, 3})
    v2 := NewVector([]float64{4, 5, 6})

    fmt.Printf("Vector addition: %v\n", v1.Add(v2).Data)
    fmt.Printf("Dot product: %.2f\n", v1.DotProduct(v2))
    fmt.Printf("Vector norm: %.2f\n", v1.Norm())
    fmt.Printf("Cosine similarity: %.2f\n", v1.CosineSimilarity(v2))
}
```

### **Matrices and Matrix Operations**

#### **Concept**

Matrices represent linear transformations, data tables, and model parameters in ML.

#### **Math Behind**

- **Matrix Multiplication**: `C = AB` where `Cᵢⱼ = Σₖ AᵢₖBₖⱼ`
- **Transpose**: `Aᵀᵢⱼ = Aⱼᵢ`
- **Determinant**: For 2x2: `det(A) = ad - bc`
- **Inverse**: `A⁻¹` such that `AA⁻¹ = I`

#### **Code Example**

```python
class MatrixOperations:
    def __init__(self):
        self.matrices = {}

    def matrix_multiply(self, A, B):
        """Multiply two matrices"""
        return np.dot(A, B)

    def matrix_transpose(self, A):
        """Transpose matrix"""
        return np.transpose(A)

    def matrix_determinant(self, A):
        """Calculate determinant of matrix"""
        return np.linalg.det(A)

    def matrix_inverse(self, A):
        """Calculate inverse of matrix"""
        return np.linalg.inv(A)

    def eigendecomposition(self, A):
        """Perform eigendecomposition"""
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors

# Example usage
matrix_ops = MatrixOperations()

# Create sample matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix multiplication:\n{matrix_ops.matrix_multiply(A, B)}")
print(f"Matrix transpose:\n{matrix_ops.matrix_transpose(A)}")
print(f"Determinant: {matrix_ops.matrix_determinant(A)}")
print(f"Inverse:\n{matrix_ops.matrix_inverse(A)}")
```

---

## 📈 **Calculus for ML**

### **Derivatives and Gradients**

#### **Concept**

Derivatives measure how a function changes with respect to its inputs, essential for optimization.

#### **Math Behind**

- **Partial Derivative**: `∂f/∂x = lim[h→0] (f(x+h) - f(x))/h`
- **Gradient**: `∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`
- **Chain Rule**: `∂f/∂x = (∂f/∂u)(∂u/∂x)`

#### **Code Example**

```python
import sympy as sp
from scipy.optimize import minimize

class CalculusOperations:
    def __init__(self):
        self.functions = {}

    def numerical_derivative(self, f, x, h=1e-5):
        """Calculate numerical derivative"""
        return (f(x + h) - f(x - h)) / (2 * h)

    def gradient_descent(self, f, grad_f, x0, learning_rate=0.01, max_iter=1000):
        """Implement gradient descent optimization"""
        x = x0.copy()
        history = [x.copy()]

        for i in range(max_iter):
            gradient = grad_f(x)
            x = x - learning_rate * gradient
            history.append(x.copy())

            if np.linalg.norm(gradient) < 1e-6:
                break

        return x, history

    def symbolic_derivative(self, expression, variable):
        """Calculate symbolic derivative"""
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)
        return sp.diff(expr, x)

# Example: Optimize quadratic function
def quadratic_function(x):
    return x[0]**2 + x[1]**2

def quadratic_gradient(x):
    return np.array([2*x[0], 2*x[1]])

calc_ops = CalculusOperations()
x0 = np.array([3.0, 4.0])
optimal_x, history = calc_ops.gradient_descent(
    quadratic_function, quadratic_gradient, x0
)

print(f"Optimal point: {optimal_x}")
print(f"Function value: {quadratic_function(optimal_x)}")
```

### **Backpropagation Mathematics**

#### **Concept**

Backpropagation uses the chain rule to compute gradients in neural networks.

#### **Math Behind**

For a neural network with loss function L:

- **Forward Pass**: `z = Wx + b`, `a = σ(z)`
- **Backward Pass**: `∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W`

#### **Code Example**

```python
class Backpropagation:
    def __init__(self):
        self.weights = {}
        self.biases = {}
        self.activations = {}

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward_pass(self, X, W1, b1, W2, b2):
        """Forward pass through neural network"""
        z1 = np.dot(X, W1) + b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.sigmoid(z2)
        return a1, a2, z1, z2

    def backward_pass(self, X, y, a1, a2, z1, z2, W1, W2):
        """Backward pass for gradient computation"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = a2 - y
        dW2 = (1/m) * np.dot(a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * self.sigmoid_derivative(z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

# Example usage
bp = Backpropagation()
# Initialize weights and biases
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Forward and backward pass
a1, a2, z1, z2 = bp.forward_pass(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = bp.backward_pass(X, y, a1, a2, z1, z2, W1, W2)

print(f"Output: {a2}")
print(f"Gradients computed successfully")
```

---

## 🎯 **Optimization Theory**

### **Gradient Descent Variants**

#### **Concept**

Different optimization algorithms for training ML models with various convergence properties.

#### **Math Behind**

- **SGD**: `θ = θ - α∇J(θ)`
- **Momentum**: `v = βv + α∇J(θ)`, `θ = θ - v`
- **Adam**: `m = β₁m + (1-β₁)∇J(θ)`, `v = β₂v + (1-β₂)(∇J(θ))²`

#### **Code Example**

```python
class Optimizers:
    def __init__(self):
        self.optimizers = {}

    def sgd(self, params, grads, learning_rate):
        """Stochastic Gradient Descent"""
        for param, grad in zip(params, grads):
            param -= learning_rate * grad
        return params

    def momentum(self, params, grads, learning_rate, momentum=0.9):
        """Momentum optimizer"""
        if not hasattr(self, 'velocity'):
            self.velocity = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = momentum * self.velocity[i] + learning_rate * grad
            param -= self.velocity[i]
        return params

    def adam(self, params, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer"""
        if not hasattr(self, 'm'):
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.t = 0

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return params

# Example usage
optimizer = Optimizers()
params = [np.random.randn(2, 3), np.random.randn(3, 1)]
grads = [np.random.randn(2, 3), np.random.randn(3, 1)]

# Test different optimizers
params_sgd = optimizer.sgd(params.copy(), grads, 0.01)
params_momentum = optimizer.momentum(params.copy(), grads, 0.01)
params_adam = optimizer.adam(params.copy(), grads, 0.01)

print("Optimizers implemented successfully")
```

---

## 📊 **Information Theory**

### **Entropy and Mutual Information**

#### **Concept**

Information theory provides measures for uncertainty and information content in data.

#### **Math Behind**

- **Entropy**: `H(X) = -Σᵢ p(xᵢ)log₂(p(xᵢ))`
- **Mutual Information**: `I(X;Y) = H(X) - H(X|Y)`
- **KL Divergence**: `D_KL(P||Q) = Σᵢ p(xᵢ)log(p(xᵢ)/q(xᵢ))`

#### **Code Example**

```python
from scipy.stats import entropy

class InformationTheory:
    def __init__(self):
        self.measures = {}

    def entropy(self, probabilities):
        """Calculate Shannon entropy"""
        # Remove zero probabilities to avoid log(0)
        probs = probabilities[probabilities > 0]
        return -np.sum(probs * np.log2(probs))

    def mutual_information(self, joint_prob, marginal_x, marginal_y):
        """Calculate mutual information"""
        mi = 0
        for i in range(len(marginal_x)):
            for j in range(len(marginal_y)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (marginal_x[i] * marginal_y[j])
                    )
        return mi

    def kl_divergence(self, p, q):
        """Calculate KL divergence"""
        # Ensure probabilities sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Remove zero probabilities
        mask = (p > 0) & (q > 0)
        p_masked = p[mask]
        q_masked = q[mask]

        return np.sum(p_masked * np.log2(p_masked / q_masked))

# Example usage
info_theory = InformationTheory()

# Sample probability distributions
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])

print(f"Entropy of p: {info_theory.entropy(p):.3f}")
print(f"KL Divergence D(p||q): {info_theory.kl_divergence(p, q):.3f}")
```

---

## 🎯 **Implementation Examples**

### **Complete ML Pipeline with Mathematical Foundations**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}

    def generate_data(self, n_samples=1000, n_features=2, n_classes=2):
        """Generate synthetic classification data"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=2,
            random_state=42
        )
        return X, y

    def normalize_features(self, X):
        """Normalize features using z-score"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std, mean, std

    def logistic_regression(self, X, y, learning_rate=0.01, max_iter=1000):
        """Implement logistic regression from scratch"""
        n_samples, n_features = X.shape

        # Initialize weights and bias
        weights = np.random.randn(n_features) * 0.01
        bias = 0

        for iteration in range(max_iter):
            # Forward pass
            z = np.dot(X, weights) + bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))

            # Compute loss (cross-entropy)
            loss = -np.mean(y * np.log(predictions + 1e-15) +
                          (1 - y) * np.log(1 - predictions + 1e-15))

            # Backward pass
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # Update parameters
            weights -= learning_rate * dw
            bias -= learning_rate * db

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

        return weights, bias

    def predict(self, X, weights, bias):
        """Make predictions using trained model"""
        z = np.dot(X, weights) + bias
        probabilities = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return (probabilities > 0.5).astype(int)

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        accuracy = np.mean(y_true == y_pred)
        return accuracy

# Example usage
pipeline = MLPipeline()

# Generate and prepare data
X, y = pipeline.generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
X_train_norm, mean, std = pipeline.normalize_features(X_train)
X_test_norm = (X_test - mean) / std

# Train model
weights, bias = pipeline.logistic_regression(X_train_norm, y_train)

# Make predictions
y_pred = pipeline.predict(X_test_norm, weights, bias)

# Evaluate
accuracy = pipeline.evaluate_model(y_test, y_pred)
print(f"Model accuracy: {accuracy:.3f}")
```

---

## 🎯 **Interview Questions**

### **Mathematical Foundations**

#### **Q1: Explain the mathematical intuition behind gradient descent**

**Answer**: Gradient descent minimizes a function by iteratively moving in the direction of steepest descent. The gradient `∇f(x)` points in the direction of maximum increase, so `-∇f(x)` points in the direction of maximum decrease. The update rule `x = x - α∇f(x)` moves the current point in the direction of steepest descent by a step size `α`.

#### **Q2: What is the chain rule and why is it important in neural networks?**

**Answer**: The chain rule allows us to compute derivatives of composite functions. In neural networks, the loss function is a composition of many functions (layers), and backpropagation uses the chain rule to compute gradients efficiently: `∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)`.

#### **Q3: Explain the mathematical relationship between eigenvalues and principal components**

**Answer**: Principal Component Analysis (PCA) finds the directions of maximum variance in data. These directions are the eigenvectors of the covariance matrix, and the corresponding eigenvalues represent the amount of variance explained by each principal component.

#### **Q4: What is the mathematical foundation of the sigmoid function?**

**Answer**: The sigmoid function `σ(x) = 1/(1 + e^(-x))` is a smooth, differentiable function that maps any real number to the interval (0,1). Its derivative is `σ'(x) = σ(x)(1 - σ(x))`, which is maximum at x=0 and decreases as |x| increases.

#### **Q5: Explain the mathematical intuition behind regularization**

**Answer**: Regularization adds a penalty term to the loss function to prevent overfitting. L1 regularization `λ||w||₁` encourages sparsity, while L2 regularization `λ||w||₂²` encourages small weights. The mathematical intuition is that we're solving a constrained optimization problem where we balance fitting the data with keeping the model simple.

### **Implementation Questions**

#### **Q6: Implement gradient descent for a quadratic function**

**Answer**: See the gradient descent implementation in the code examples above.

#### **Q7: Write a function to compute the Jacobian matrix**

**Answer**:

```python
def compute_jacobian(f, x, h=1e-5):
    """Compute Jacobian matrix numerically"""
    n = len(x)
    jacobian = np.zeros((n, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        jacobian[:, i] = (f(x_plus) - f(x_minus)) / (2 * h)

    return jacobian
```

#### **Q8: Implement matrix factorization using SVD**

**Answer**:

```python
def matrix_factorization_svd(A, k):
    """Matrix factorization using SVD"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Keep only top k components
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct matrix
    A_reconstructed = U_k @ np.diag(s_k) @ Vt_k

    return U_k, s_k, Vt_k, A_reconstructed
```

---

## 🚀 **Next Steps**

1. **Practice**: Implement all mathematical concepts from scratch
2. **Visualize**: Create plots and diagrams to understand concepts
3. **Apply**: Use mathematical foundations in ML algorithms
4. **Optimize**: Focus on computational efficiency and numerical stability
5. **Interview**: Practice mathematical ML interview questions

---

**Ready to dive deeper into ML algorithms? Let's move to the Machine Learning section!** 🎯
