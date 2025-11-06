---
# Auto-generated front matter
Title: Logisticregression
LastUpdated: 2025-11-06T20:45:59.089694
Tags: []
Status: draft
---

# Logistic Regression in JavaScript

## Overview

Logistic Regression is a statistical method for analyzing datasets in which there are one or more independent variables that determine an outcome. It's used for binary classification problems where the outcome is categorical.

## Mathematical Foundation

### Sigmoid Function
The sigmoid function maps any real-valued number into a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- β₀, β₁, ..., βₙ are the model parameters
- x₁, x₂, ..., xₙ are the input features

### Cost Function
The cost function for logistic regression is the log-likelihood:

```
J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

Where:
- m = number of training examples
- y = actual output (0 or 1)
- h(x) = predicted probability

## Implementation

### Basic Logistic Regression Class
```javascript
class LogisticRegression {
  constructor(learningRate = 0.01, maxIterations = 1000) {
    this.learningRate = learningRate;
    this.maxIterations = maxIterations;
    this.weights = null;
    this.bias = 0;
    this.costHistory = [];
  }

  // Sigmoid activation function
  sigmoid(z) {
    // Prevent overflow
    z = Math.max(-500, Math.min(500, z));
    return 1 / (1 + Math.exp(-z));
  }

  // Initialize weights
  initializeWeights(numFeatures) {
    this.weights = new Array(numFeatures).fill(0).map(() => 
      (Math.random() - 0.5) * 0.01
    );
  }

  // Forward propagation
  forward(X) {
    const predictions = [];
    
    for (let i = 0; i < X.length; i++) {
      let z = this.bias;
      
      for (let j = 0; j < X[i].length; j++) {
        z += this.weights[j] * X[i][j];
      }
      
      predictions.push(this.sigmoid(z));
    }
    
    return predictions;
  }

  // Compute cost
  computeCost(y, predictions) {
    let cost = 0;
    const m = y.length;
    
    for (let i = 0; i < m; i++) {
      const pred = Math.max(1e-15, Math.min(1 - 1e-15, predictions[i]));
      cost += y[i] * Math.log(pred) + (1 - y[i]) * Math.log(1 - pred);
    }
    
    return -cost / m;
  }

  // Compute gradients
  computeGradients(X, y, predictions) {
    const m = X.length;
    const dw = new Array(this.weights.length).fill(0);
    let db = 0;
    
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      
      for (let j = 0; j < this.weights.length; j++) {
        dw[j] += error * X[i][j];
      }
      
      db += error;
    }
    
    // Average gradients
    for (let j = 0; j < dw.length; j++) {
      dw[j] /= m;
    }
    db /= m;
    
    return { dw, db };
  }

  // Update parameters
  updateParameters(gradients) {
    for (let j = 0; j < this.weights.length; j++) {
      this.weights[j] -= this.learningRate * gradients.dw[j];
    }
    this.bias -= this.learningRate * gradients.db;
  }

  // Train the model
  fit(X, y) {
    if (X.length !== y.length) {
      throw new Error('X and y must have the same length');
    }
    
    this.initializeWeights(X[0].length);
    
    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      // Forward propagation
      const predictions = this.forward(X);
      
      // Compute cost
      const cost = this.computeCost(y, predictions);
      this.costHistory.push(cost);
      
      // Compute gradients
      const gradients = this.computeGradients(X, y, predictions);
      
      // Update parameters
      this.updateParameters(gradients);
      
      // Early stopping
      if (iteration > 0 && Math.abs(this.costHistory[iteration - 1] - cost) < 1e-6) {
        console.log(`Converged at iteration ${iteration}`);
        break;
      }
    }
    
    return this;
  }

  // Make predictions
  predict(X) {
    const probabilities = this.forward(X);
    return probabilities.map(p => p >= 0.5 ? 1 : 0);
  }

  // Get prediction probabilities
  predictProba(X) {
    return this.forward(X);
  }

  // Calculate accuracy
  score(X, y) {
    const predictions = this.predict(X);
    let correct = 0;
    
    for (let i = 0; i < y.length; i++) {
      if (predictions[i] === y[i]) {
        correct++;
      }
    }
    
    return correct / y.length;
  }
}
```

### Advanced Logistic Regression with Regularization
```javascript
class RegularizedLogisticRegression extends LogisticRegression {
  constructor(learningRate = 0.01, maxIterations = 1000, lambda = 0.1, regularization = 'l2') {
    super(learningRate, maxIterations);
    this.lambda = lambda;
    this.regularization = regularization; // 'l1' or 'l2'
  }

  // Compute cost with regularization
  computeCost(y, predictions) {
    const m = y.length;
    let cost = 0;
    
    // Cross-entropy cost
    for (let i = 0; i < m; i++) {
      const pred = Math.max(1e-15, Math.min(1 - 1e-15, predictions[i]));
      cost += y[i] * Math.log(pred) + (1 - y[i]) * Math.log(1 - pred);
    }
    cost = -cost / m;
    
    // Add regularization
    if (this.regularization === 'l2') {
      const regCost = this.lambda * this.weights.reduce((sum, w) => sum + w * w, 0) / (2 * m);
      cost += regCost;
    } else if (this.regularization === 'l1') {
      const regCost = this.lambda * this.weights.reduce((sum, w) => sum + Math.abs(w), 0) / m;
      cost += regCost;
    }
    
    return cost;
  }

  // Compute gradients with regularization
  computeGradients(X, y, predictions) {
    const m = X.length;
    const dw = new Array(this.weights.length).fill(0);
    let db = 0;
    
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      
      for (let j = 0; j < this.weights.length; j++) {
        dw[j] += error * X[i][j];
      }
      
      db += error;
    }
    
    // Average gradients
    for (let j = 0; j < dw.length; j++) {
      dw[j] /= m;
    }
    db /= m;
    
    // Add regularization gradients
    if (this.regularization === 'l2') {
      for (let j = 0; j < dw.length; j++) {
        dw[j] += (this.lambda * this.weights[j]) / m;
      }
    } else if (this.regularization === 'l1') {
      for (let j = 0; j < dw.length; j++) {
        dw[j] += (this.lambda * Math.sign(this.weights[j])) / m;
      }
    }
    
    return { dw, db };
  }
}
```

### Feature Scaling and Preprocessing
```javascript
class FeatureScaler {
  constructor() {
    this.mean = null;
    this.std = null;
    this.min = null;
    this.max = null;
  }

  // Standardization (Z-score normalization)
  fitStandardScaler(X) {
    const numFeatures = X[0].length;
    this.mean = new Array(numFeatures).fill(0);
    this.std = new Array(numFeatures).fill(0);
    
    // Calculate mean
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < numFeatures; j++) {
        this.mean[j] += X[i][j];
      }
    }
    
    for (let j = 0; j < numFeatures; j++) {
      this.mean[j] /= X.length;
    }
    
    // Calculate standard deviation
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < numFeatures; j++) {
        this.std[j] += Math.pow(X[i][j] - this.mean[j], 2);
      }
    }
    
    for (let j = 0; j < numFeatures; j++) {
      this.std[j] = Math.sqrt(this.std[j] / X.length);
    }
  }

  transformStandardScaler(X) {
    const scaledX = [];
    
    for (let i = 0; i < X.length; i++) {
      const scaledRow = [];
      for (let j = 0; j < X[i].length; j++) {
        scaledRow.push((X[i][j] - this.mean[j]) / this.std[j]);
      }
      scaledX.push(scaledRow);
    }
    
    return scaledX;
  }

  // Min-Max normalization
  fitMinMaxScaler(X) {
    const numFeatures = X[0].length;
    this.min = new Array(numFeatures).fill(Infinity);
    this.max = new Array(numFeatures).fill(-Infinity);
    
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < numFeatures; j++) {
        this.min[j] = Math.min(this.min[j], X[i][j]);
        this.max[j] = Math.max(this.max[j], X[i][j]);
      }
    }
  }

  transformMinMaxScaler(X) {
    const scaledX = [];
    
    for (let i = 0; i < X.length; i++) {
      const scaledRow = [];
      for (let j = 0; j < X[i].length; j++) {
        const scaled = (X[i][j] - this.min[j]) / (this.max[j] - this.min[j]);
        scaledRow.push(scaled);
      }
      scaledX.push(scaledRow);
    }
    
    return scaledX;
  }
}
```

### Model Evaluation and Metrics
```javascript
class ModelEvaluator {
  constructor() {}

  // Confusion Matrix
  confusionMatrix(yTrue, yPred) {
    const matrix = {
      truePositive: 0,
      trueNegative: 0,
      falsePositive: 0,
      falseNegative: 0
    };
    
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === 1 && yPred[i] === 1) {
        matrix.truePositive++;
      } else if (yTrue[i] === 0 && yPred[i] === 0) {
        matrix.trueNegative++;
      } else if (yTrue[i] === 0 && yPred[i] === 1) {
        matrix.falsePositive++;
      } else if (yTrue[i] === 1 && yPred[i] === 0) {
        matrix.falseNegative++;
      }
    }
    
    return matrix;
  }

  // Accuracy
  accuracy(yTrue, yPred) {
    const matrix = this.confusionMatrix(yTrue, yPred);
    return (matrix.truePositive + matrix.trueNegative) / yTrue.length;
  }

  // Precision
  precision(yTrue, yPred) {
    const matrix = this.confusionMatrix(yTrue, yPred);
    return matrix.truePositive / (matrix.truePositive + matrix.falsePositive);
  }

  // Recall
  recall(yTrue, yPred) {
    const matrix = this.confusionMatrix(yTrue, yPred);
    return matrix.truePositive / (matrix.truePositive + matrix.falseNegative);
  }

  // F1 Score
  f1Score(yTrue, yPred) {
    const prec = this.precision(yTrue, yPred);
    const rec = this.recall(yTrue, yPred);
    return 2 * (prec * rec) / (prec + rec);
  }

  // ROC AUC Score
  rocAucScore(yTrue, yProba) {
    const thresholds = [...new Set(yProba)].sort((a, b) => b - a);
    let auc = 0;
    
    for (let i = 0; i < thresholds.length - 1; i++) {
      const threshold = thresholds[i];
      const yPred = yProba.map(p => p >= threshold ? 1 : 0);
      const matrix = this.confusionMatrix(yTrue, yPred);
      
      const tpr = matrix.truePositive / (matrix.truePositive + matrix.falseNegative);
      const fpr = matrix.falsePositive / (matrix.falsePositive + matrix.trueNegative);
      
      auc += tpr * (fpr - (matrix.falsePositive / (matrix.falsePositive + matrix.trueNegative)));
    }
    
    return auc;
  }
}
```

### Complete Example with Dataset
```javascript
// Example usage with a simple dataset
function exampleUsage() {
  // Sample dataset (features: [age, income], target: [0, 1])
  const X = [
    [25, 40000],
    [30, 50000],
    [35, 60000],
    [40, 70000],
    [45, 80000],
    [50, 90000],
    [55, 100000],
    [60, 110000]
  ];
  
  const y = [0, 0, 0, 1, 1, 1, 1, 1]; // 0: no purchase, 1: purchase
  
  // Feature scaling
  const scaler = new FeatureScaler();
  scaler.fitStandardScaler(X);
  const XScaled = scaler.transformStandardScaler(X);
  
  // Split data (simple split for demonstration)
  const trainSize = Math.floor(XScaled.length * 0.7);
  const XTrain = XScaled.slice(0, trainSize);
  const yTrain = y.slice(0, trainSize);
  const XTest = XScaled.slice(trainSize);
  const yTest = y.slice(trainSize);
  
  // Train model
  const model = new LogisticRegression(0.1, 1000);
  model.fit(XTrain, yTrain);
  
  // Make predictions
  const yPred = model.predict(XTest);
  const yProba = model.predictProba(XTest);
  
  // Evaluate model
  const evaluator = new ModelEvaluator();
  console.log('Accuracy:', evaluator.accuracy(yTest, yPred));
  console.log('Precision:', evaluator.precision(yTest, yPred));
  console.log('Recall:', evaluator.recall(yTest, yPred));
  console.log('F1 Score:', evaluator.f1Score(yTest, yPred));
  
  return {
    model,
    predictions: yPred,
    probabilities: yProba,
    metrics: {
      accuracy: evaluator.accuracy(yTest, yPred),
      precision: evaluator.precision(yTest, yPred),
      recall: evaluator.recall(yTest, yPred),
      f1Score: evaluator.f1Score(yTest, yPred)
    }
  };
}

// Run example
const result = exampleUsage();
console.log('Model trained successfully!');
console.log('Metrics:', result.metrics);
```

## Key Features

### Mathematical Concepts
- **Sigmoid Function**: Maps real values to probabilities
- **Cross-Entropy Loss**: Measures prediction accuracy
- **Gradient Descent**: Optimizes model parameters
- **Regularization**: Prevents overfitting

### Implementation Features
- **Feature Scaling**: Standardization and normalization
- **Regularization**: L1 and L2 regularization
- **Model Evaluation**: Comprehensive metrics
- **Early Stopping**: Prevents overfitting
- **Batch Processing**: Efficient training

### Use Cases
- **Binary Classification**: Yes/No predictions
- **Probability Estimation**: Confidence scores
- **Feature Importance**: Weight analysis
- **Medical Diagnosis**: Disease prediction
- **Marketing**: Customer behavior prediction

## Extension Ideas

### Advanced Features
1. **Multi-class Classification**: One-vs-Rest approach
2. **Stochastic Gradient Descent**: Online learning
3. **Feature Engineering**: Polynomial features
4. **Cross-Validation**: Model validation
5. **Hyperparameter Tuning**: Grid search

### Production Features
1. **Model Persistence**: Save/load models
2. **Real-time Prediction**: Streaming predictions
3. **A/B Testing**: Model comparison
4. **Monitoring**: Performance tracking
5. **API Integration**: RESTful endpoints
