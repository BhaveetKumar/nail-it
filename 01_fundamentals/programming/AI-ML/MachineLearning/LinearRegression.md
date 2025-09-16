# 📈 Linear Regression - JavaScript Implementation

> **Comprehensive guide to Linear Regression with JavaScript/Node.js implementations**

## 🎯 **Overview**

Linear Regression is a fundamental machine learning algorithm used for predicting continuous values. This guide covers the mathematical foundations, implementation in JavaScript, and practical applications for backend developers.

## 📚 **Table of Contents**

1. [Mathematical Foundations](#mathematical-foundations/)
2. [JavaScript Implementation](#javascript-implementation/)
3. [Gradient Descent](#gradient-descent/)
4. [Regularization](#regularization/)
5. [Performance Metrics](#performance-metrics/)
6. [Real-world Applications](#real-world-applications/)
7. [API Integration](#api-integration/)

---

## 🧮 **Mathematical Foundations**

### **Linear Regression Formula**

Linear regression models the relationship between a dependent variable (y) and one or more independent variables (x) using a linear equation:

**Simple Linear Regression:**
```
y = mx + b
```

**Multiple Linear Regression:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` = dependent variable (target)
- `x₁, x₂, ..., xₙ` = independent variables (features)
- `β₀` = intercept (bias)
- `β₁, β₂, ..., βₙ` = coefficients (weights)
- `ε` = error term

### **Cost Function (Mean Squared Error)**

```
J(θ) = (1/2m) * Σ(hθ(xⁱ) - yⁱ)²
```

Where:
- `m` = number of training examples
- `hθ(xⁱ)` = predicted value
- `yⁱ` = actual value

---

## 💻 **JavaScript Implementation**

### **Basic Linear Regression Class**

```javascript
class LinearRegression {
    constructor(learningRate = 0.01, maxIterations = 1000) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.weights = null;
        this.bias = 0;
        this.costHistory = [];
    }
    
    // Initialize weights
    initializeWeights(numFeatures) {
        this.weights = new Array(numFeatures).fill(0);
        this.bias = 0;
    }
    
    // Predict function
    predict(X) {
        if (!this.weights) {
            throw new Error('Model not trained yet');
        }
        
        if (Array.isArray(X[0])) {
            // Multiple samples
            return X.map(sample => this.predictSingle(sample));
        } else {
            // Single sample
            return this.predictSingle(X);
        }
    }
    
    predictSingle(sample) {
        let prediction = this.bias;
        for (let i = 0; i < sample.length; i++) {
            prediction += this.weights[i] * sample[i];
        }
        return prediction;
    }
    
    // Cost function
    computeCost(X, y) {
        const m = X.length;
        let totalCost = 0;
        
        for (let i = 0; i < m; i++) {
            const prediction = this.predictSingle(X[i]);
            const error = prediction - y[i];
            totalCost += error * error;
        }
        
        return totalCost / (2 * m);
    }
    
    // Gradient descent
    gradientDescent(X, y) {
        const m = X.length;
        const n = X[0].length;
        
        // Initialize weights if not already done
        if (!this.weights) {
            this.initializeWeights(n);
        }
        
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Compute predictions
            const predictions = this.predict(X);
            
            // Compute gradients
            const weightGradients = new Array(n).fill(0);
            let biasGradient = 0;
            
            for (let i = 0; i < m; i++) {
                const error = predictions[i] - y[i];
                biasGradient += error;
                
                for (let j = 0; j < n; j++) {
                    weightGradients[j] += error * X[i][j];
                }
            }
            
            // Update parameters
            this.bias -= (this.learningRate * biasGradient) / m;
            for (let j = 0; j < n; j++) {
                this.weights[j] -= (this.learningRate * weightGradients[j]) / m;
            }
            
            // Record cost
            const cost = this.computeCost(X, y);
            this.costHistory.push(cost);
            
            // Early stopping if cost doesn't improve
            if (iteration > 0 && Math.abs(this.costHistory[iteration - 1] - cost) < 1e-6) {
                console.log(`Converged after ${iteration + 1} iterations`);
                break;
            }
        }
    }
    
    // Train the model
    fit(X, y) {
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        
        if (X.length === 0) {
            throw new Error('Training data cannot be empty');
        }
        
        // Normalize features
        this.normalizeFeatures(X);
        
        // Train using gradient descent
        this.gradientDescent(X, y);
        
        console.log('Training completed');
        console.log(`Final cost: ${this.costHistory[this.costHistory.length - 1]}`);
    }
    
    // Feature normalization
    normalizeFeatures(X) {
        const n = X[0].length;
        this.featureMeans = new Array(n);
        this.featureStds = new Array(n);
        
        // Calculate means and standard deviations
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            this.featureMeans[j] = sum / X.length;
            
            let variance = 0;
            for (let i = 0; i < X.length; i++) {
                variance += Math.pow(X[i][j] - this.featureMeans[j], 2);
            }
            this.featureStds[j] = Math.sqrt(variance / X.length);
            
            // Avoid division by zero
            if (this.featureStds[j] === 0) {
                this.featureStds[j] = 1;
            }
        }
        
        // Normalize features
        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < n; j++) {
                X[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
            }
        }
    }
    
    // Normalize new data using training statistics
    normalizeNewData(X) {
        if (!this.featureMeans || !this.featureStds) {
            throw new Error('Model must be trained first');
        }
        
        const normalizedX = X.map(sample => 
            sample.map((value, index) => 
                (value - this.featureMeans[index]) / this.featureStds[index]
            )
        );
        
        return normalizedX;
    }
    
    // Get model parameters
    getParameters() {
        return {
            weights: this.weights,
            bias: this.bias,
            featureMeans: this.featureMeans,
            featureStds: this.featureStds
        };
    }
    
    // Set model parameters
    setParameters(parameters) {
        this.weights = parameters.weights;
        this.bias = parameters.bias;
        this.featureMeans = parameters.featureMeans;
        this.featureStds = parameters.featureStds;
    }
}
```

### **Advanced Linear Regression with Matrix Operations**

```javascript
class MatrixLinearRegression {
    constructor(learningRate = 0.01, maxIterations = 1000) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.theta = null;
        this.costHistory = [];
    }
    
    // Matrix operations
    matrixMultiply(A, B) {
        const rows = A.length;
        const cols = B[0].length;
        const result = Array(rows).fill().map(() => Array(cols).fill(0));
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                for (let k = 0; k < A[0].length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    matrixTranspose(matrix) {
        return matrix[0].map((_, colIndex) => 
            matrix.map(row => row[colIndex])
        );
    }
    
    matrixSubtract(A, B) {
        return A.map((row, i) => 
            row.map((val, j) => val - B[i][j])
        );
    }
    
    matrixScalarMultiply(matrix, scalar) {
        return matrix.map(row => 
            row.map(val => val * scalar)
        );
    }
    
    // Normal equation (closed-form solution)
    normalEquation(X, y) {
        // Add bias column (column of ones)
        const XWithBias = X.map(row => [1, ...row]);
        
        // Calculate (X^T * X)^-1 * X^T * y
        const XTranspose = this.matrixTranspose(XWithBias);
        const XTX = this.matrixMultiply(XTranspose, XWithBias);
        const XTXInverse = this.matrixInverse(XTX);
        const XTy = this.matrixMultiply(XTranspose, y.map(val => [val]));
        
        this.theta = this.matrixMultiply(XTXInverse, XTy);
        
        return this.theta;
    }
    
    // Matrix inverse using Gaussian elimination
    matrixInverse(matrix) {
        const n = matrix.length;
        const identity = Array(n).fill().map((_, i) => 
            Array(n).fill().map((_, j) => i === j ? 1 : 0)
        );
        
        // Create augmented matrix
        const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
        
        // Forward elimination
        for (let i = 0; i < n; i++) {
            // Find pivot
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
            
            // Make diagonal element 1
            const pivot = augmented[i][i];
            for (let j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate column
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = augmented[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        return augmented.map(row => row.slice(n));
    }
    
    // Predict using matrix operations
    predict(X) {
        if (!this.theta) {
            throw new Error('Model not trained yet');
        }
        
        const XWithBias = X.map(row => [1, ...row]);
        const predictions = this.matrixMultiply(XWithBias, this.theta);
        
        return predictions.map(row => row[0]);
    }
    
    // Train using normal equation
    fit(X, y) {
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        
        this.normalEquation(X, y);
        console.log('Training completed using normal equation');
    }
}
```

---

## 📊 **Performance Metrics**

### **Regression Metrics Implementation**

```javascript
class RegressionMetrics {
    // Mean Squared Error
    static meanSquaredError(yTrue, yPred) {
        if (yTrue.length !== yPred.length) {
            throw new Error('Arrays must have the same length');
        }
        
        let sum = 0;
        for (let i = 0; i < yTrue.length; i++) {
            sum += Math.pow(yTrue[i] - yPred[i], 2);
        }
        
        return sum / yTrue.length;
    }
    
    // Root Mean Squared Error
    static rootMeanSquaredError(yTrue, yPred) {
        return Math.sqrt(this.meanSquaredError(yTrue, yPred));
    }
    
    // Mean Absolute Error
    static meanAbsoluteError(yTrue, yPred) {
        if (yTrue.length !== yPred.length) {
            throw new Error('Arrays must have the same length');
        }
        
        let sum = 0;
        for (let i = 0; i < yTrue.length; i++) {
            sum += Math.abs(yTrue[i] - yPred[i]);
        }
        
        return sum / yTrue.length;
    }
    
    // R-squared (Coefficient of Determination)
    static rSquared(yTrue, yPred) {
        if (yTrue.length !== yPred.length) {
            throw new Error('Arrays must have the same length');
        }
        
        const yMean = yTrue.reduce((sum, val) => sum + val, 0) / yTrue.length;
        
        let ssRes = 0; // Sum of squares of residuals
        let ssTot = 0; // Total sum of squares
        
        for (let i = 0; i < yTrue.length; i++) {
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
            ssTot += Math.pow(yTrue[i] - yMean, 2);
        }
        
        return 1 - (ssRes / ssTot);
    }
    
    // Adjusted R-squared
    static adjustedRSquared(yTrue, yPred, numFeatures) {
        const n = yTrue.length;
        const r2 = this.rSquared(yTrue, yPred);
        
        return 1 - (1 - r2) * (n - 1) / (n - numFeatures - 1);
    }
    
    // Mean Absolute Percentage Error
    static meanAbsolutePercentageError(yTrue, yPred) {
        if (yTrue.length !== yPred.length) {
            throw new Error('Arrays must have the same length');
        }
        
        let sum = 0;
        for (let i = 0; i < yTrue.length; i++) {
            if (yTrue[i] !== 0) {
                sum += Math.abs((yTrue[i] - yPred[i]) / yTrue[i]);
            }
        }
        
        return (sum / yTrue.length) * 100;
    }
    
    // Comprehensive evaluation
    static evaluate(yTrue, yPred, numFeatures = null) {
        const metrics = {
            mse: this.meanSquaredError(yTrue, yPred),
            rmse: this.rootMeanSquaredError(yTrue, yPred),
            mae: this.meanAbsoluteError(yTrue, yPred),
            r2: this.rSquared(yTrue, yPred),
            mape: this.meanAbsolutePercentageError(yTrue, yPred)
        };
        
        if (numFeatures !== null) {
            metrics.adjustedR2 = this.adjustedRSquared(yTrue, yPred, numFeatures);
        }
        
        return metrics;
    }
}
```

---

## 🔧 **Regularization**

### **Ridge Regression (L2 Regularization)**

```javascript
class RidgeRegression {
    constructor(alpha = 1.0, learningRate = 0.01, maxIterations = 1000) {
        this.alpha = alpha; // Regularization strength
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.weights = null;
        this.bias = 0;
        this.costHistory = [];
    }
    
    // Ridge regression cost function
    computeCost(X, y) {
        const m = X.length;
        let mse = 0;
        let regularization = 0;
        
        // Mean squared error
        for (let i = 0; i < m; i++) {
            const prediction = this.predictSingle(X[i]);
            const error = prediction - y[i];
            mse += error * error;
        }
        mse /= (2 * m);
        
        // L2 regularization term
        for (let j = 0; j < this.weights.length; j++) {
            regularization += this.weights[j] * this.weights[j];
        }
        regularization *= this.alpha / (2 * m);
        
        return mse + regularization;
    }
    
    // Gradient descent with L2 regularization
    gradientDescent(X, y) {
        const m = X.length;
        const n = X[0].length;
        
        if (!this.weights) {
            this.initializeWeights(n);
        }
        
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            const predictions = this.predict(X);
            
            // Compute gradients
            const weightGradients = new Array(n).fill(0);
            let biasGradient = 0;
            
            for (let i = 0; i < m; i++) {
                const error = predictions[i] - y[i];
                biasGradient += error;
                
                for (let j = 0; j < n; j++) {
                    weightGradients[j] += error * X[i][j];
                }
            }
            
            // Update parameters with L2 regularization
            this.bias -= (this.learningRate * biasGradient) / m;
            for (let j = 0; j < n; j++) {
                // Add L2 regularization term to gradient
                const regularizationGradient = this.alpha * this.weights[j] / m;
                this.weights[j] -= (this.learningRate * (weightGradients[j] / m + regularizationGradient));
            }
            
            // Record cost
            const cost = this.computeCost(X, y);
            this.costHistory.push(cost);
            
            // Early stopping
            if (iteration > 0 && Math.abs(this.costHistory[iteration - 1] - cost) < 1e-6) {
                console.log(`Converged after ${iteration + 1} iterations`);
                break;
            }
        }
    }
    
    // Inherit other methods from LinearRegression
    initializeWeights(numFeatures) {
        this.weights = new Array(numFeatures).fill(0);
        this.bias = 0;
    }
    
    predictSingle(sample) {
        let prediction = this.bias;
        for (let i = 0; i < sample.length; i++) {
            prediction += this.weights[i] * sample[i];
        }
        return prediction;
    }
    
    predict(X) {
        if (!this.weights) {
            throw new Error('Model not trained yet');
        }
        
        if (Array.isArray(X[0])) {
            return X.map(sample => this.predictSingle(sample));
        } else {
            return this.predictSingle(X);
        }
    }
    
    fit(X, y) {
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        
        this.normalizeFeatures(X);
        this.gradientDescent(X, y);
        console.log('Ridge regression training completed');
    }
    
    normalizeFeatures(X) {
        const n = X[0].length;
        this.featureMeans = new Array(n);
        this.featureStds = new Array(n);
        
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            this.featureMeans[j] = sum / X.length;
            
            let variance = 0;
            for (let i = 0; i < X.length; i++) {
                variance += Math.pow(X[i][j] - this.featureMeans[j], 2);
            }
            this.featureStds[j] = Math.sqrt(variance / X.length);
            
            if (this.featureStds[j] === 0) {
                this.featureStds[j] = 1;
            }
        }
        
        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < n; j++) {
                X[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
            }
        }
    }
}
```

### **Lasso Regression (L1 Regularization)**

```javascript
class LassoRegression {
    constructor(alpha = 1.0, learningRate = 0.01, maxIterations = 1000) {
        this.alpha = alpha;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.weights = null;
        this.bias = 0;
        this.costHistory = [];
    }
    
    // Lasso regression cost function
    computeCost(X, y) {
        const m = X.length;
        let mse = 0;
        let regularization = 0;
        
        // Mean squared error
        for (let i = 0; i < m; i++) {
            const prediction = this.predictSingle(X[i]);
            const error = prediction - y[i];
            mse += error * error;
        }
        mse /= (2 * m);
        
        // L1 regularization term
        for (let j = 0; j < this.weights.length; j++) {
            regularization += Math.abs(this.weights[j]);
        }
        regularization *= this.alpha / m;
        
        return mse + regularization;
    }
    
    // Coordinate descent for Lasso
    coordinateDescent(X, y) {
        const m = X.length;
        const n = X[0].length;
        
        if (!this.weights) {
            this.initializeWeights(n);
        }
        
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Update bias
            let biasSum = 0;
            for (let i = 0; i < m; i++) {
                let prediction = this.bias;
                for (let j = 0; j < n; j++) {
                    prediction += this.weights[j] * X[i][j];
                }
                biasSum += (y[i] - prediction);
            }
            this.bias = biasSum / m;
            
            // Update each weight using coordinate descent
            for (let j = 0; j < n; j++) {
                let numerator = 0;
                let denominator = 0;
                
                for (let i = 0; i < m; i++) {
                    let residual = y[i] - this.bias;
                    for (let k = 0; k < n; k++) {
                        if (k !== j) {
                            residual -= this.weights[k] * X[i][k];
                        }
                    }
                    
                    numerator += X[i][j] * residual;
                    denominator += X[i][j] * X[i][j];
                }
                
                // Soft thresholding for L1 regularization
                const threshold = this.alpha / (2 * denominator);
                const rawWeight = numerator / denominator;
                
                if (rawWeight > threshold) {
                    this.weights[j] = rawWeight - threshold;
                } else if (rawWeight < -threshold) {
                    this.weights[j] = rawWeight + threshold;
                } else {
                    this.weights[j] = 0;
                }
            }
            
            // Record cost
            const cost = this.computeCost(X, y);
            this.costHistory.push(cost);
            
            // Early stopping
            if (iteration > 0 && Math.abs(this.costHistory[iteration - 1] - cost) < 1e-6) {
                console.log(`Converged after ${iteration + 1} iterations`);
                break;
            }
        }
    }
    
    // Inherit other methods
    initializeWeights(numFeatures) {
        this.weights = new Array(numFeatures).fill(0);
        this.bias = 0;
    }
    
    predictSingle(sample) {
        let prediction = this.bias;
        for (let i = 0; i < sample.length; i++) {
            prediction += this.weights[i] * sample[i];
        }
        return prediction;
    }
    
    predict(X) {
        if (!this.weights) {
            throw new Error('Model not trained yet');
        }
        
        if (Array.isArray(X[0])) {
            return X.map(sample => this.predictSingle(sample));
        } else {
            return this.predictSingle(X);
        }
    }
    
    fit(X, y) {
        if (X.length !== y.length) {
            throw new Error('X and y must have the same length');
        }
        
        this.normalizeFeatures(X);
        this.coordinateDescent(X, y);
        console.log('Lasso regression training completed');
    }
    
    normalizeFeatures(X) {
        const n = X[0].length;
        this.featureMeans = new Array(n);
        this.featureStds = new Array(n);
        
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            this.featureMeans[j] = sum / X.length;
            
            let variance = 0;
            for (let i = 0; i < X.length; i++) {
                variance += Math.pow(X[i][j] - this.featureMeans[j], 2);
            }
            this.featureStds[j] = Math.sqrt(variance / X.length);
            
            if (this.featureStds[j] === 0) {
                this.featureStds[j] = 1;
            }
        }
        
        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < n; j++) {
                X[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
            }
        }
    }
}
```

---

## 🌐 **API Integration**

### **Express.js API for Linear Regression**

```javascript
const express = require('express');
const cors = require('cors');
const { LinearRegression, RidgeRegression, LassoRegression } = require('./linear-regression');

class LinearRegressionAPI {
    constructor() {
        this.app = express();
        this.models = new Map();
        this.setupMiddleware();
        this.setupRoutes();
    }
    
    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
    }
    
    setupRoutes() {
        // Train model
        this.app.post('/api/train', this.trainModel.bind(this));
        
        // Predict
        this.app.post('/api/predict', this.predict.bind(this));
        
        // Get model info
        this.app.get('/api/model/:id', this.getModelInfo.bind(this));
        
        // List models
        this.app.get('/api/models', this.listModels.bind(this));
        
        // Delete model
        this.app.delete('/api/model/:id', this.deleteModel.bind(this));
        
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ status: 'healthy', timestamp: new Date() });
        });
    }
    
    async trainModel(req, res) {
        try {
            const { 
                modelType = 'linear', 
                X, 
                y, 
                learningRate = 0.01, 
                maxIterations = 1000,
                alpha = 1.0 
            } = req.body;
            
            if (!X || !y) {
                return res.status(400).json({ 
                    error: 'X and y are required' 
                });
            }
            
            if (X.length !== y.length) {
                return res.status(400).json({ 
                    error: 'X and y must have the same length' 
                });
            }
            
            // Create model based on type
            let model;
            switch (modelType) {
                case 'ridge':
                    model = new RidgeRegression(alpha, learningRate, maxIterations);
                    break;
                case 'lasso':
                    model = new LassoRegression(alpha, learningRate, maxIterations);
                    break;
                default:
                    model = new LinearRegression(learningRate, maxIterations);
            }
            
            // Train model
            model.fit(X, y);
            
            // Generate model ID
            const modelId = this.generateModelId();
            
            // Store model
            this.models.set(modelId, {
                model,
                modelType,
                trainingData: { X, y },
                createdAt: new Date(),
                parameters: model.getParameters()
            });
            
            res.json({
                success: true,
                modelId,
                modelType,
                costHistory: model.costHistory,
                finalCost: model.costHistory[model.costHistory.length - 1]
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: error.message 
            });
        }
    }
    
    async predict(req, res) {
        try {
            const { modelId, X } = req.body;
            
            if (!modelId || !X) {
                return res.status(400).json({ 
                    error: 'modelId and X are required' 
                });
            }
            
            const modelData = this.models.get(modelId);
            if (!modelData) {
                return res.status(404).json({ 
                    error: 'Model not found' 
                });
            }
            
            const predictions = modelData.model.predict(X);
            
            res.json({
                success: true,
                predictions,
                modelId,
                modelType: modelData.modelType
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: error.message 
            });
        }
    }
    
    async getModelInfo(req, res) {
        try {
            const { id } = req.params;
            
            const modelData = this.models.get(id);
            if (!modelData) {
                return res.status(404).json({ 
                    error: 'Model not found' 
                });
            }
            
            res.json({
                success: true,
                modelId: id,
                modelType: modelData.modelType,
                createdAt: modelData.createdAt,
                parameters: modelData.parameters,
                costHistory: modelData.model.costHistory
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: error.message 
            });
        }
    }
    
    async listModels(req, res) {
        try {
            const models = Array.from(this.models.entries()).map(([id, data]) => ({
                modelId: id,
                modelType: data.modelType,
                createdAt: data.createdAt,
                finalCost: data.model.costHistory[data.model.costHistory.length - 1]
            }));
            
            res.json({
                success: true,
                models,
                count: models.length
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: error.message 
            });
        }
    }
    
    async deleteModel(req, res) {
        try {
            const { id } = req.params;
            
            if (!this.models.has(id)) {
                return res.status(404).json({ 
                    error: 'Model not found' 
                });
            }
            
            this.models.delete(id);
            
            res.json({
                success: true,
                message: 'Model deleted successfully'
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: error.message 
            });
        }
    }
    
    generateModelId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    start(port = 3000) {
        this.app.listen(port, () => {
            console.log(`Linear Regression API server running on port ${port}`);
        });
    }
}

// Usage example
if (require.main === module) {
    const api = new LinearRegressionAPI();
    api.start(3000);
}

module.exports = { LinearRegressionAPI };
```

---

## 🎯 **Key Takeaways**

### **Mathematical Concepts**
- Linear regression models linear relationships between variables
- Cost function measures prediction accuracy
- Gradient descent optimizes model parameters
- Regularization prevents overfitting

### **Implementation Features**
- Feature normalization for better convergence
- Multiple optimization algorithms
- Regularization techniques (Ridge, Lasso)
- Comprehensive performance metrics

### **API Integration**
- RESTful API for model training and prediction
- Support for multiple model types
- Model persistence and management
- Error handling and validation

### **Best Practices**
- Always normalize features
- Use cross-validation for model evaluation
- Apply regularization to prevent overfitting
- Monitor training progress with cost history

---

**🎉 This comprehensive guide provides everything needed to implement and use Linear Regression in JavaScript/Node.js applications!**
