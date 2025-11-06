---
# Auto-generated front matter
Title: Gradient Descent
LastUpdated: 2025-11-06T20:45:58.317707
Tags: []
Status: draft
---

# Gradient Descent - Optimization Algorithm

## Overview

Gradient Descent is an iterative optimization algorithm used to minimize a cost function by finding the optimal parameters. It's the foundation of many machine learning algorithms and is used to train neural networks, linear regression, and other models.

## Key Concepts

- **Cost Function**: Function to minimize (e.g., Mean Squared Error)
- **Gradient**: Partial derivatives of the cost function
- **Learning Rate**: Step size for parameter updates
- **Convergence**: When the algorithm reaches the minimum
- **Local vs Global Minimum**: Different types of minima

## Gradient Descent Types

### 1. Batch Gradient Descent
- Uses entire dataset for each update
- More stable but slower
- Better for convex functions

### 2. Stochastic Gradient Descent (SGD)
- Uses one sample at a time
- Faster but more noisy
- Better for non-convex functions

### 3. Mini-batch Gradient Descent
- Uses small batches of data
- Balance between stability and speed
- Most commonly used in practice

## Mathematical Foundation

The gradient descent algorithm updates parameters using:

```
θ = θ - α * ∇J(θ)
```

Where:
- θ: parameters
- α: learning rate
- ∇J(θ): gradient of cost function

## Go Implementation

```go
package main

import (
    "fmt"
    "log"
    "math"
    "math/rand"
    "time"
)

// Point represents a data point
type Point struct {
    X float64
    Y float64
}

// LinearRegression represents a linear regression model
type LinearRegression struct {
    Weights []float64
    Bias    float64
    LearningRate float64
    Epochs  int
}

// NewLinearRegression creates a new linear regression model
func NewLinearRegression(learningRate float64, epochs int) *LinearRegression {
    return &LinearRegression{
        LearningRate: learningRate,
        Epochs:       epochs,
    }
}

// Fit trains the model using gradient descent
func (lr *LinearRegression) Fit(X [][]float64, y []float64) {
    if len(X) == 0 || len(X[0]) == 0 {
        log.Fatal("Empty dataset")
    }
    
    // Initialize weights and bias
    nFeatures := len(X[0])
    lr.Weights = make([]float64, nFeatures)
    lr.Bias = 0.0
    
    // Initialize weights with small random values
    rand.Seed(time.Now().UnixNano())
    for i := range lr.Weights {
        lr.Weights[i] = rand.Float64() * 0.01
    }
    
    // Gradient descent
    for epoch := 0; epoch < lr.Epochs; epoch++ {
        lr.updateWeights(X, y)
        
        if epoch%100 == 0 {
            cost := lr.computeCost(X, y)
            fmt.Printf("Epoch %d, Cost: %.6f\n", epoch, cost)
        }
    }
}

// updateWeights updates weights using gradient descent
func (lr *LinearRegression) updateWeights(X [][]float64, y []float64) {
    nSamples := len(X)
    nFeatures := len(X[0])
    
    // Compute gradients
    weightGradients := make([]float64, nFeatures)
    biasGradient := 0.0
    
    for i := 0; i < nSamples; i++ {
        prediction := lr.predict(X[i])
        error := prediction - y[i]
        
        // Update weight gradients
        for j := 0; j < nFeatures; j++ {
            weightGradients[j] += error * X[i][j]
        }
        
        // Update bias gradient
        biasGradient += error
    }
    
    // Average gradients
    for j := 0; j < nFeatures; j++ {
        weightGradients[j] /= float64(nSamples)
    }
    biasGradient /= float64(nSamples)
    
    // Update weights and bias
    for j := 0; j < nFeatures; j++ {
        lr.Weights[j] -= lr.LearningRate * weightGradients[j]
    }
    lr.Bias -= lr.LearningRate * biasGradient
}

// computeCost computes the mean squared error
func (lr *LinearRegression) computeCost(X [][]float64, y []float64) float64 {
    nSamples := len(X)
    totalError := 0.0
    
    for i := 0; i < nSamples; i++ {
        prediction := lr.predict(X[i])
        error := prediction - y[i]
        totalError += error * error
    }
    
    return totalError / (2.0 * float64(nSamples))
}

// predict makes a prediction for a single sample
func (lr *LinearRegression) predict(x []float64) float64 {
    prediction := lr.Bias
    
    for i := 0; i < len(x); i++ {
        prediction += lr.Weights[i] * x[i]
    }
    
    return prediction
}

// Predict makes predictions for multiple samples
func (lr *LinearRegression) Predict(X [][]float64) []float64 {
    predictions := make([]float64, len(X))
    
    for i, x := range X {
        predictions[i] = lr.predict(x)
    }
    
    return predictions
}

// R2Score computes the R-squared score
func (lr *LinearRegression) R2Score(X [][]float64, y []float64) float64 {
    predictions := lr.Predict(X)
    
    // Compute mean of actual values
    meanY := 0.0
    for _, val := range y {
        meanY += val
    }
    meanY /= float64(len(y))
    
    // Compute total sum of squares
    tss := 0.0
    for _, val := range y {
        tss += (val - meanY) * (val - meanY)
    }
    
    // Compute residual sum of squares
    rss := 0.0
    for i, val := range y {
        rss += (val - predictions[i]) * (val - predictions[i])
    }
    
    return 1.0 - (rss / tss)
}

// StochasticGradientDescent implements SGD
func (lr *LinearRegression) StochasticGradientDescent(X [][]float64, y []float64) {
    nSamples := len(X)
    nFeatures := len(X[0])
    
    // Initialize weights and bias
    lr.Weights = make([]float64, nFeatures)
    lr.Bias = 0.0
    
    // Initialize weights with small random values
    rand.Seed(time.Now().UnixNano())
    for i := range lr.Weights {
        lr.Weights[i] = rand.Float64() * 0.01
    }
    
    // SGD
    for epoch := 0; epoch < lr.Epochs; epoch++ {
        // Shuffle data
        indices := make([]int, nSamples)
        for i := range indices {
            indices[i] = i
        }
        rand.Shuffle(nSamples, func(i, j int) {
            indices[i], indices[j] = indices[j], indices[i]
        })
        
        // Update weights for each sample
        for _, idx := range indices {
            lr.updateWeightsSGD(X[idx], y[idx])
        }
        
        if epoch%100 == 0 {
            cost := lr.computeCost(X, y)
            fmt.Printf("Epoch %d, Cost: %.6f\n", epoch, cost)
        }
    }
}

// updateWeightsSGD updates weights using SGD
func (lr *LinearRegression) updateWeightsSGD(x []float64, y float64) {
    prediction := lr.predict(x)
    error := prediction - y
    
    // Update weights
    for i := 0; i < len(x); i++ {
        lr.Weights[i] -= lr.LearningRate * error * x[i]
    }
    
    // Update bias
    lr.Bias -= lr.LearningRate * error
}

// MiniBatchGradientDescent implements mini-batch GD
func (lr *LinearRegression) MiniBatchGradientDescent(X [][]float64, y []float64, batchSize int) {
    nSamples := len(X)
    nFeatures := len(X[0])
    
    // Initialize weights and bias
    lr.Weights = make([]float64, nFeatures)
    lr.Bias = 0.0
    
    // Initialize weights with small random values
    rand.Seed(time.Now().UnixNano())
    for i := range lr.Weights {
        lr.Weights[i] = rand.Float64() * 0.01
    }
    
    // Mini-batch GD
    for epoch := 0; epoch < lr.Epochs; epoch++ {
        // Shuffle data
        indices := make([]int, nSamples)
        for i := range indices {
            indices[i] = i
        }
        rand.Shuffle(nSamples, func(i, j int) {
            indices[i], indices[j] = indices[j], indices[i]
        })
        
        // Process mini-batches
        for i := 0; i < nSamples; i += batchSize {
            end := i + batchSize
            if end > nSamples {
                end = nSamples
            }
            
            batchIndices := indices[i:end]
            lr.updateWeightsMiniBatch(X, y, batchIndices)
        }
        
        if epoch%100 == 0 {
            cost := lr.computeCost(X, y)
            fmt.Printf("Epoch %d, Cost: %.6f\n", epoch, cost)
        }
    }
}

// updateWeightsMiniBatch updates weights using mini-batch
func (lr *LinearRegression) updateWeightsMiniBatch(X [][]float64, y []float64, batchIndices []int) {
    nFeatures := len(X[0])
    batchSize := len(batchIndices)
    
    // Compute gradients
    weightGradients := make([]float64, nFeatures)
    biasGradient := 0.0
    
    for _, idx := range batchIndices {
        prediction := lr.predict(X[idx])
        error := prediction - y[idx]
        
        // Update weight gradients
        for j := 0; j < nFeatures; j++ {
            weightGradients[j] += error * X[idx][j]
        }
        
        // Update bias gradient
        biasGradient += error
    }
    
    // Average gradients
    for j := 0; j < nFeatures; j++ {
        weightGradients[j] /= float64(batchSize)
    }
    biasGradient /= float64(batchSize)
    
    // Update weights and bias
    for j := 0; j < nFeatures; j++ {
        lr.Weights[j] -= lr.LearningRate * weightGradients[j]
    }
    lr.Bias -= lr.LearningRate * biasGradient
}

// Example usage
func main() {
    // Generate sample data
    X, y := generateSampleData(100, 2)
    
    // Create and train model
    lr := NewLinearRegression(0.01, 1000)
    
    fmt.Println("Training with Batch Gradient Descent:")
    lr.Fit(X, y)
    
    // Make predictions
    predictions := lr.Predict(X)
    
    // Compute R2 score
    r2 := lr.R2Score(X, y)
    fmt.Printf("R2 Score: %.4f\n", r2)
    
    // Test SGD
    fmt.Println("\nTraining with Stochastic Gradient Descent:")
    lr2 := NewLinearRegression(0.01, 1000)
    lr2.StochasticGradientDescent(X, y)
    
    r2SGD := lr2.R2Score(X, y)
    fmt.Printf("R2 Score (SGD): %.4f\n", r2SGD)
    
    // Test Mini-batch GD
    fmt.Println("\nTraining with Mini-batch Gradient Descent:")
    lr3 := NewLinearRegression(0.01, 1000)
    lr3.MiniBatchGradientDescent(X, y, 32)
    
    r2MiniBatch := lr3.R2Score(X, y)
    fmt.Printf("R2 Score (Mini-batch): %.4f\n", r2MiniBatch)
}

// generateSampleData generates sample data for testing
func generateSampleData(nSamples, nFeatures int) ([][]float64, []float64) {
    X := make([][]float64, nSamples)
    y := make([]float64, nSamples)
    
    rand.Seed(time.Now().UnixNano())
    
    // True weights and bias
    trueWeights := make([]float64, nFeatures)
    for i := range trueWeights {
        trueWeights[i] = rand.Float64() * 2 - 1
    }
    trueBias := rand.Float64() * 2 - 1
    
    for i := 0; i < nSamples; i++ {
        X[i] = make([]float64, nFeatures)
        for j := 0; j < nFeatures; j++ {
            X[i][j] = rand.Float64() * 10 - 5
        }
        
        // Compute true y value
        y[i] = trueBias
        for j := 0; j < nFeatures; j++ {
            y[i] += trueWeights[j] * X[i][j]
        }
        
        // Add noise
        y[i] += rand.Float64() * 0.1
    }
    
    return X, y
}
```

## Node.js Implementation

```javascript
class LinearRegression {
  constructor(learningRate, epochs) {
    this.learningRate = learningRate;
    this.epochs = epochs;
    this.weights = [];
    this.bias = 0;
  }

  fit(X, y) {
    if (X.length === 0 || X[0].length === 0) {
      throw new Error("Empty dataset");
    }

    // Initialize weights and bias
    const nFeatures = X[0].length;
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    // Initialize weights with small random values
    for (let i = 0; i < nFeatures; i++) {
      this.weights[i] = (Math.random() - 0.5) * 0.01;
    }

    // Gradient descent
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      this.updateWeights(X, y);

      if (epoch % 100 === 0) {
        const cost = this.computeCost(X, y);
        console.log(`Epoch ${epoch}, Cost: ${cost.toFixed(6)}`);
      }
    }
  }

  updateWeights(X, y) {
    const nSamples = X.length;
    const nFeatures = X[0].length;

    // Compute gradients
    const weightGradients = new Array(nFeatures).fill(0);
    let biasGradient = 0;

    for (let i = 0; i < nSamples; i++) {
      const prediction = this.predict(X[i]);
      const error = prediction - y[i];

      // Update weight gradients
      for (let j = 0; j < nFeatures; j++) {
        weightGradients[j] += error * X[i][j];
      }

      // Update bias gradient
      biasGradient += error;
    }

    // Average gradients
    for (let j = 0; j < nFeatures; j++) {
      weightGradients[j] /= nSamples;
    }
    biasGradient /= nSamples;

    // Update weights and bias
    for (let j = 0; j < nFeatures; j++) {
      this.weights[j] -= this.learningRate * weightGradients[j];
    }
    this.bias -= this.learningRate * biasGradient;
  }

  computeCost(X, y) {
    const nSamples = X.length;
    let totalError = 0;

    for (let i = 0; i < nSamples; i++) {
      const prediction = this.predict(X[i]);
      const error = prediction - y[i];
      totalError += error * error;
    }

    return totalError / (2 * nSamples);
  }

  predict(x) {
    let prediction = this.bias;

    for (let i = 0; i < x.length; i++) {
      prediction += this.weights[i] * x[i];
    }

    return prediction;
  }

  predictBatch(X) {
    const predictions = [];

    for (let i = 0; i < X.length; i++) {
      predictions.push(this.predict(X[i]));
    }

    return predictions;
  }

  r2Score(X, y) {
    const predictions = this.predictBatch(X);

    // Compute mean of actual values
    const meanY = y.reduce((sum, val) => sum + val, 0) / y.length;

    // Compute total sum of squares
    const tss = y.reduce((sum, val) => sum + (val - meanY) ** 2, 0);

    // Compute residual sum of squares
    const rss = y.reduce((sum, val, i) => sum + (val - predictions[i]) ** 2, 0);

    return 1 - (rss / tss);
  }

  stochasticGradientDescent(X, y) {
    const nSamples = X.length;
    const nFeatures = X[0].length;

    // Initialize weights and bias
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    // Initialize weights with small random values
    for (let i = 0; i < nFeatures; i++) {
      this.weights[i] = (Math.random() - 0.5) * 0.01;
    }

    // SGD
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle data
      const indices = Array.from({ length: nSamples }, (_, i) => i);
      for (let i = nSamples - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Update weights for each sample
      for (const idx of indices) {
        this.updateWeightsSGD(X[idx], y[idx]);
      }

      if (epoch % 100 === 0) {
        const cost = this.computeCost(X, y);
        console.log(`Epoch ${epoch}, Cost: ${cost.toFixed(6)}`);
      }
    }
  }

  updateWeightsSGD(x, y) {
    const prediction = this.predict(x);
    const error = prediction - y;

    // Update weights
    for (let i = 0; i < x.length; i++) {
      this.weights[i] -= this.learningRate * error * x[i];
    }

    // Update bias
    this.bias -= this.learningRate * error;
  }

  miniBatchGradientDescent(X, y, batchSize) {
    const nSamples = X.length;
    const nFeatures = X[0].length;

    // Initialize weights and bias
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    // Initialize weights with small random values
    for (let i = 0; i < nFeatures; i++) {
      this.weights[i] = (Math.random() - 0.5) * 0.01;
    }

    // Mini-batch GD
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle data
      const indices = Array.from({ length: nSamples }, (_, i) => i);
      for (let i = nSamples - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Process mini-batches
      for (let i = 0; i < nSamples; i += batchSize) {
        const end = Math.min(i + batchSize, nSamples);
        const batchIndices = indices.slice(i, end);
        this.updateWeightsMiniBatch(X, y, batchIndices);
      }

      if (epoch % 100 === 0) {
        const cost = this.computeCost(X, y);
        console.log(`Epoch ${epoch}, Cost: ${cost.toFixed(6)}`);
      }
    }
  }

  updateWeightsMiniBatch(X, y, batchIndices) {
    const nFeatures = X[0].length;
    const batchSize = batchIndices.length;

    // Compute gradients
    const weightGradients = new Array(nFeatures).fill(0);
    let biasGradient = 0;

    for (const idx of batchIndices) {
      const prediction = this.predict(X[idx]);
      const error = prediction - y[idx];

      // Update weight gradients
      for (let j = 0; j < nFeatures; j++) {
        weightGradients[j] += error * X[idx][j];
      }

      // Update bias gradient
      biasGradient += error;
    }

    // Average gradients
    for (let j = 0; j < nFeatures; j++) {
      weightGradients[j] /= batchSize;
    }
    biasGradient /= batchSize;

    // Update weights and bias
    for (let j = 0; j < nFeatures; j++) {
      this.weights[j] -= this.learningRate * weightGradients[j];
    }
    this.bias -= this.learningRate * biasGradient;
  }
}

// Example usage
function main() {
  // Generate sample data
  const { X, y } = generateSampleData(100, 2);

  // Create and train model
  const lr = new LinearRegression(0.01, 1000);

  console.log("Training with Batch Gradient Descent:");
  lr.fit(X, y);

  // Make predictions
  const predictions = lr.predictBatch(X);

  // Compute R2 score
  const r2 = lr.r2Score(X, y);
  console.log(`R2 Score: ${r2.toFixed(4)}`);

  // Test SGD
  console.log("\nTraining with Stochastic Gradient Descent:");
  const lr2 = new LinearRegression(0.01, 1000);
  lr2.stochasticGradientDescent(X, y);

  const r2SGD = lr2.r2Score(X, y);
  console.log(`R2 Score (SGD): ${r2SGD.toFixed(4)}`);

  // Test Mini-batch GD
  console.log("\nTraining with Mini-batch Gradient Descent:");
  const lr3 = new LinearRegression(0.01, 1000);
  lr3.miniBatchGradientDescent(X, y, 32);

  const r2MiniBatch = lr3.r2Score(X, y);
  console.log(`R2 Score (Mini-batch): ${r2MiniBatch.toFixed(4)}`);
}

// Generate sample data for testing
function generateSampleData(nSamples, nFeatures) {
  const X = [];
  const y = [];

  // True weights and bias
  const trueWeights = [];
  for (let i = 0; i < nFeatures; i++) {
    trueWeights[i] = (Math.random() - 0.5) * 2;
  }
  const trueBias = (Math.random() - 0.5) * 2;

  for (let i = 0; i < nSamples; i++) {
    const x = [];
    for (let j = 0; j < nFeatures; j++) {
      x[j] = (Math.random() - 0.5) * 10;
    }
    X.push(x);

    // Compute true y value
    let yVal = trueBias;
    for (let j = 0; j < nFeatures; j++) {
      yVal += trueWeights[j] * x[j];
    }

    // Add noise
    yVal += (Math.random() - 0.5) * 0.1;
    y.push(yVal);
  }

  return { X, y };
}

if (require.main === module) {
  main();
}
```

## Benefits

1. **Universal**: Works for many optimization problems
2. **Simple**: Easy to understand and implement
3. **Efficient**: Can handle large datasets
4. **Flexible**: Can be adapted for different problems

## Trade-offs

1. **Learning Rate**: Sensitive to learning rate choice
2. **Local Minima**: May get stuck in local minima
3. **Convergence**: May not converge for some functions
4. **Computational Cost**: Can be expensive for large datasets

## Use Cases

- **Linear Regression**: Fitting linear models
- **Neural Networks**: Training deep learning models
- **Logistic Regression**: Classification problems
- **Support Vector Machines**: Finding optimal hyperplanes

## Best Practices

1. **Learning Rate**: Start with 0.01 and adjust
2. **Feature Scaling**: Normalize features before training
3. **Regularization**: Use L1/L2 regularization to prevent overfitting
4. **Early Stopping**: Stop training when validation error stops improving
5. **Momentum**: Use momentum to speed up convergence

## Common Pitfalls

1. **Learning Rate Too High**: May overshoot the minimum
2. **Learning Rate Too Low**: May take too long to converge
3. **Feature Scaling**: Not scaling features can cause convergence issues
4. **Local Minima**: May get stuck in local minima

## Interview Questions

1. **What's the difference between batch and stochastic gradient descent?**
   - Batch uses entire dataset, SGD uses one sample at a time

2. **How do you choose the learning rate?**
   - Start with 0.01, use grid search or adaptive methods

3. **What happens if the learning rate is too high?**
   - May overshoot the minimum and diverge

4. **How do you handle local minima?**
   - Use momentum, different initialization, or different algorithms

## Time Complexity

- **Batch GD**: O(n * m * epochs) where n is samples, m is features
- **SGD**: O(m * epochs) per sample
- **Mini-batch GD**: O(batch_size * m * epochs) per batch

## Space Complexity

- **Storage**: O(m) where m is number of features
- **Auxiliary Space**: O(1) for basic operations

The optimal solution uses:
1. **Proper Learning Rate**: Choose appropriate learning rate
2. **Feature Scaling**: Normalize features before training
3. **Regularization**: Use regularization to prevent overfitting
4. **Early Stopping**: Stop training when appropriate
