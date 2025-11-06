---
# Auto-generated front matter
Title: Calculus
LastUpdated: 2025-11-06T20:45:58.416211
Tags: []
Status: draft
---

# Calculus for Engineers

## Table of Contents

1. [Overview](#overview)
2. [Limits and Continuity](#limits-and-continuity)
3. [Derivatives](#derivatives)
4. [Integrals](#integrals)
5. [Optimization](#optimization)
6. [Applications](#applications)
7. [Implementations](#implementations)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)
10. [Projects](#projects)

## Overview

### Learning Objectives

- Master limits and continuity concepts
- Understand derivatives and their applications
- Learn integration techniques and applications
- Apply calculus to optimization problems
- Use calculus in machine learning and engineering

### What is Calculus?

Calculus is the mathematical study of continuous change, focusing on derivatives (rates of change) and integrals (accumulation of quantities). It's essential for optimization, machine learning, physics simulations, and many engineering applications.

## Limits and Continuity

### 1. Limit Definition and Computation

#### Numerical Limit Computation

```go
package main

import (
    "fmt"
    "math"
)

type Function func(float64) float64

// Example functions
func polynomial(x float64) float64 {
    return x*x - 2*x + 1
}

func rational(x float64) float64 {
    if math.Abs(x-1) < 1e-10 {
        return math.NaN() // Avoid division by zero
    }
    return (x*x - 1) / (x - 1)
}

func trigonometric(x float64) float64 {
    if math.Abs(x) < 1e-10 {
        return 1.0 // sin(x)/x approaches 1 as x approaches 0
    }
    return math.Sin(x) / x
}

func exponential(x float64) float64 {
    return math.Exp(x)
}

func ComputeLimit(f Function, point float64, direction string) float64 {
    var h float64
    if direction == "left" {
        h = -1e-10
    } else if direction == "right" {
        h = 1e-10
    } else {
        h = 1e-10
    }
    
    // Use numerical approximation
    return f(point + h)
}

func ComputeLimitNumerical(f Function, point float64, tolerance float64) float64 {
    h := 1e-6
    prevValue := f(point + h)
    
    for i := 0; i < 20; i++ {
        h /= 2
        currentValue := f(point + h)
        
        if math.Abs(currentValue - prevValue) < tolerance {
            return currentValue
        }
        
        prevValue = currentValue
    }
    
    return prevValue
}

func IsContinuous(f Function, point float64) bool {
    // Check if limit exists and equals function value
    leftLimit := ComputeLimit(f, point, "left")
    rightLimit := ComputeLimit(f, point, "right")
    functionValue := f(point)
    
    return !math.IsNaN(leftLimit) && !math.IsNaN(rightLimit) && 
           math.Abs(leftLimit-rightLimit) < 1e-10 &&
           math.Abs(leftLimit-functionValue) < 1e-10
}

// Example usage
func main() {
    fmt.Println("Polynomial at x=2:", polynomial(2))
    fmt.Println("Limit of (x²-1)/(x-1) as x→1:", ComputeLimitNumerical(rational, 1, 1e-10))
    fmt.Println("Limit of sin(x)/x as x→0:", ComputeLimitNumerical(trigonometric, 0, 1e-10))
    fmt.Println("Is polynomial continuous at x=2:", IsContinuous(polynomial, 2))
}
```

#### Node.js Implementation

```javascript
class Calculus {
    // Example functions
    static polynomial(x) {
        return x * x - 2 * x + 1;
    }
    
    static rational(x) {
        if (Math.abs(x - 1) < 1e-10) {
            return NaN; // Avoid division by zero
        }
        return (x * x - 1) / (x - 1);
    }
    
    static trigonometric(x) {
        if (Math.abs(x) < 1e-10) {
            return 1.0; // sin(x)/x approaches 1 as x approaches 0
        }
        return Math.sin(x) / x;
    }
    
    static exponential(x) {
        return Math.exp(x);
    }
    
    static computeLimit(f, point, direction = 'right') {
        let h;
        if (direction === 'left') {
            h = -1e-10;
        } else if (direction === 'right') {
            h = 1e-10;
        } else {
            h = 1e-10;
        }
        
        return f(point + h);
    }
    
    static computeLimitNumerical(f, point, tolerance = 1e-10) {
        let h = 1e-6;
        let prevValue = f(point + h);
        
        for (let i = 0; i < 20; i++) {
            h /= 2;
            const currentValue = f(point + h);
            
            if (Math.abs(currentValue - prevValue) < tolerance) {
                return currentValue;
            }
            
            prevValue = currentValue;
        }
        
        return prevValue;
    }
    
    static isContinuous(f, point) {
        const leftLimit = this.computeLimit(f, point, 'left');
        const rightLimit = this.computeLimit(f, point, 'right');
        const functionValue = f(point);
        
        return !isNaN(leftLimit) && !isNaN(rightLimit) && 
               Math.abs(leftLimit - rightLimit) < 1e-10 &&
               Math.abs(leftLimit - functionValue) < 1e-10;
    }
}

// Example usage
console.log('Polynomial at x=2:', Calculus.polynomial(2));
console.log('Limit of (x²-1)/(x-1) as x→1:', Calculus.computeLimitNumerical(Calculus.rational, 1));
console.log('Limit of sin(x)/x as x→0:', Calculus.computeLimitNumerical(Calculus.trigonometric, 0));
console.log('Is polynomial continuous at x=2:', Calculus.isContinuous(Calculus.polynomial, 2));
```

## Derivatives

### 1. Numerical Differentiation

#### Forward, Backward, and Central Differences

```go
package main

import (
    "fmt"
    "math"
)

type DerivativeCalculator struct {
    h float64
}

func NewDerivativeCalculator(h float64) *DerivativeCalculator {
    return &DerivativeCalculator{h: h}
}

func (dc *DerivativeCalculator) ForwardDifference(f Function, x float64) float64 {
    return (f(x + dc.h) - f(x)) / dc.h
}

func (dc *DerivativeCalculator) BackwardDifference(f Function, x float64) float64 {
    return (f(x) - f(x - dc.h)) / dc.h
}

func (dc *DerivativeCalculator) CentralDifference(f Function, x float64) float64 {
    return (f(x + dc.h) - f(x - dc.h)) / (2 * dc.h)
}

func (dc *DerivativeCalculator) SecondDerivative(f Function, x float64) float64 {
    return (f(x + dc.h) - 2*f(x) + f(x - dc.h)) / (dc.h * dc.h)
}

func (dc *DerivativeCalculator) PartialDerivative(f func(float64, float64) float64, x, y float64, variable string) float64 {
    if variable == "x" {
        return (f(x + dc.h, y) - f(x - dc.h, y)) / (2 * dc.h)
    } else if variable == "y" {
        return (f(x, y + dc.h) - f(x, y - dc.h)) / (2 * dc.h)
    }
    return 0
}

// Example functions
func quadratic(x float64) float64 {
    return x*x + 2*x + 1
}

func cubic(x float64) float64 {
    return x*x*x - 3*x*x + 2*x + 1
}

func sine(x float64) float64 {
    return math.Sin(x)
}

func cosine(x float64) float64 {
    return math.Cos(x)
}

func exponential(x float64) float64 {
    return math.Exp(x)
}

func logarithm(x float64) float64 {
    if x <= 0 {
        return math.NaN()
    }
    return math.Log(x)
}

// Multivariable function
func polynomial2D(x, y float64) float64 {
    return x*x + 2*x*y + y*y
}

func main() {
    calc := NewDerivativeCalculator(1e-6)
    
    fmt.Println("Derivatives of x² + 2x + 1 at x=2:")
    fmt.Println("Forward difference:", calc.ForwardDifference(quadratic, 2))
    fmt.Println("Backward difference:", calc.BackwardDifference(quadratic, 2))
    fmt.Println("Central difference:", calc.CentralDifference(quadratic, 2))
    fmt.Println("Analytical result: 2x + 2 =", 2*2 + 2)
    
    fmt.Println("\nSecond derivative of x³ - 3x² + 2x + 1 at x=1:")
    fmt.Println("Numerical:", calc.SecondDerivative(cubic, 1))
    fmt.Println("Analytical: 6x - 6 =", 6*1 - 6)
    
    fmt.Println("\nPartial derivatives of x² + 2xy + y² at (1, 2):")
    fmt.Println("∂f/∂x:", calc.PartialDerivative(polynomial2D, 1, 2, "x"))
    fmt.Println("∂f/∂y:", calc.PartialDerivative(polynomial2D, 1, 2, "y"))
}
```

### 2. Gradient and Jacobian

#### Multivariable Calculus

```go
package main

import "math"

type VectorFunction func([]float64) []float64

type GradientCalculator struct {
    h float64
}

func NewGradientCalculator(h float64) *GradientCalculator {
    return &GradientCalculator{h: h}
}

func (gc *GradientCalculator) Gradient(f func([]float64) float64, x []float64) []float64 {
    n := len(x)
    gradient := make([]float64, n)
    
    for i := 0; i < n; i++ {
        xPlus := make([]float64, n)
        xMinus := make([]float64, n)
        copy(xPlus, x)
        copy(xMinus, x)
        
        xPlus[i] += gc.h
        xMinus[i] -= gc.h
        
        gradient[i] = (f(xPlus) - f(xMinus)) / (2 * gc.h)
    }
    
    return gradient
}

func (gc *GradientCalculator) Jacobian(f VectorFunction, x []float64) [][]float64 {
    n := len(x)
    m := len(f(x))
    
    jacobian := make([][]float64, m)
    for i := 0; i < m; i++ {
        jacobian[i] = make([]float64, n)
    }
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            xPlus := make([]float64, n)
            xMinus := make([]float64, n)
            copy(xPlus, x)
            copy(xMinus, x)
            
            xPlus[j] += gc.h
            xMinus[j] -= gc.h
            
            fPlus := f(xPlus)
            fMinus := f(xMinus)
            
            jacobian[i][j] = (fPlus[i] - fMinus[i]) / (2 * gc.h)
        }
    }
    
    return jacobian
}

func (gc *GradientCalculator) Hessian(f func([]float64) float64, x []float64) [][]float64 {
    n := len(x)
    hessian := make([][]float64, n)
    for i := 0; i < n; i++ {
        hessian[i] = make([]float64, n)
    }
    
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            xPlusPlus := make([]float64, n)
            xPlusMinus := make([]float64, n)
            xMinusPlus := make([]float64, n)
            xMinusMinus := make([]float64, n)
            
            copy(xPlusPlus, x)
            copy(xPlusMinus, x)
            copy(xMinusPlus, x)
            copy(xMinusMinus, x)
            
            xPlusPlus[i] += gc.h
            xPlusPlus[j] += gc.h
            xPlusMinus[i] += gc.h
            xPlusMinus[j] -= gc.h
            xMinusPlus[i] -= gc.h
            xMinusPlus[j] += gc.h
            xMinusMinus[i] -= gc.h
            xMinusMinus[j] -= gc.h
            
            hessian[i][j] = (f(xPlusPlus) - f(xPlusMinus) - f(xMinusPlus) + f(xMinusMinus)) / (4 * gc.h * gc.h)
        }
    }
    
    return hessian
}

// Example functions
func rosenbrock(x []float64) float64 {
    if len(x) < 2 {
        return 0
    }
    a := 1.0
    b := 100.0
    return (a-x[0])*(a-x[0]) + b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
}

func sphere(x []float64) float64 {
    sum := 0.0
    for _, xi := range x {
        sum += xi * xi
    }
    return sum
}

func main() {
    calc := NewGradientCalculator(1e-6)
    
    x := []float64{1.0, 1.0}
    
    fmt.Println("Gradient of Rosenbrock function at (1, 1):")
    gradient := calc.Gradient(rosenbrock, x)
    for i, g := range gradient {
        fmt.Printf("∂f/∂x%d = %.6f\n", i+1, g)
    }
    
    fmt.Println("\nHessian of Rosenbrock function at (1, 1):")
    hessian := calc.Hessian(rosenbrock, x)
    for i, row := range hessian {
        fmt.Printf("Row %d: %v\n", i+1, row)
    }
}
```

## Integrals

### 1. Numerical Integration

#### Trapezoidal and Simpson's Rules

```go
package main

import (
    "fmt"
    "math"
)

type Integrator struct {
    tolerance float64
    maxIterations int
}

func NewIntegrator(tolerance float64, maxIterations int) *Integrator {
    return &Integrator{
        tolerance: tolerance,
        maxIterations: maxIterations,
    }
}

func (intg *Integrator) TrapezoidalRule(f Function, a, b float64, n int) float64 {
    h := (b - a) / float64(n)
    sum := (f(a) + f(b)) / 2.0
    
    for i := 1; i < n; i++ {
        x := a + float64(i)*h
        sum += f(x)
    }
    
    return sum * h
}

func (intg *Integrator) SimpsonsRule(f Function, a, b float64, n int) float64 {
    if n%2 != 0 {
        n++ // Ensure n is even
    }
    
    h := (b - a) / float64(n)
    sum := f(a) + f(b)
    
    for i := 1; i < n; i++ {
        x := a + float64(i)*h
        if i%2 == 0 {
            sum += 2 * f(x)
        } else {
            sum += 4 * f(x)
        }
    }
    
    return sum * h / 3.0
}

func (intg *Integrator) AdaptiveSimpson(f Function, a, b float64) float64 {
    return intg.adaptiveSimpsonRecursive(f, a, b, f(a), f(b), f((a+b)/2), 0)
}

func (intg *Integrator) adaptiveSimpsonRecursive(f Function, a, b, fa, fb, fc float64, depth int) float64 {
    if depth > intg.maxIterations {
        return intg.SimpsonsRule(f, a, b, 2)
    }
    
    c := (a + b) / 2
    fd := f((a + c) / 2)
    fe := f((c + b) / 2)
    
    // Simpson's rule for [a, b]
    s1 := (b - a) / 6 * (fa + 4*fc + fb)
    
    // Simpson's rule for [a, c] + [c, b]
    s2 := (c - a) / 6 * (fa + 4*fd + fc) + (b - c) / 6 * (fc + 4*fe + fb)
    
    if math.Abs(s1 - s2) < intg.tolerance {
        return s2
    }
    
    return intg.adaptiveSimpsonRecursive(f, a, c, fa, fc, fd, depth+1) +
           intg.adaptiveSimpsonRecursive(f, c, b, fc, fb, fe, depth+1)
}

func (intg *Integrator) MonteCarlo(f Function, a, b float64, n int) float64 {
    sum := 0.0
    for i := 0; i < n; i++ {
        x := a + (b-a)*float64(i)/float64(n)
        sum += f(x)
    }
    return sum * (b - a) / float64(n)
}

// Example functions
func polynomial(x float64) float64 {
    return x*x + 2*x + 1
}

func sine(x float64) float64 {
    return math.Sin(x)
}

func exponential(x float64) float64 {
    return math.Exp(x)
}

func gaussian(x float64) float64 {
    return math.Exp(-x*x)
}

func main() {
    integrator := NewIntegrator(1e-10, 20)
    
    fmt.Println("Integrating x² + 2x + 1 from 0 to 2:")
    fmt.Println("Trapezoidal (n=100):", integrator.TrapezoidalRule(polynomial, 0, 2, 100))
    fmt.Println("Simpson's (n=100):", integrator.SimpsonsRule(polynomial, 0, 2, 100))
    fmt.Println("Adaptive Simpson:", integrator.AdaptiveSimpson(polynomial, 0, 2))
    fmt.Println("Analytical result: 26/3 =", 26.0/3.0)
    
    fmt.Println("\nIntegrating sin(x) from 0 to π:")
    fmt.Println("Trapezoidal (n=100):", integrator.TrapezoidalRule(sine, 0, math.Pi, 100))
    fmt.Println("Simpson's (n=100):", integrator.SimpsonsRule(sine, 0, math.Pi, 100))
    fmt.Println("Adaptive Simpson:", integrator.AdaptiveSimpson(sine, 0, math.Pi))
    fmt.Println("Analytical result: 2 =", 2.0)
}
```

## Optimization

### 1. Gradient Descent

#### Unconstrained Optimization

```go
package main

import (
    "fmt"
    "math"
)

type Optimizer struct {
    learningRate float64
    tolerance float64
    maxIterations int
}

func NewOptimizer(learningRate, tolerance float64, maxIterations int) *Optimizer {
    return &Optimizer{
        learningRate: learningRate,
        tolerance: tolerance,
        maxIterations: maxIterations,
    }
}

func (opt *Optimizer) GradientDescent(f func([]float64) float64, gradient func([]float64) []float64, x0 []float64) ([]float64, int) {
    x := make([]float64, len(x0))
    copy(x, x0)
    
    for i := 0; i < opt.maxIterations; i++ {
        grad := gradient(x)
        
        // Check convergence
        maxGrad := 0.0
        for _, g := range grad {
            if math.Abs(g) > maxGrad {
                maxGrad = math.Abs(g)
            }
        }
        
        if maxGrad < opt.tolerance {
            return x, i
        }
        
        // Update parameters
        for j := 0; j < len(x); j++ {
            x[j] -= opt.learningRate * grad[j]
        }
    }
    
    return x, opt.maxIterations
}

func (opt *Optimizer) NewtonMethod(f func([]float64) float64, gradient func([]float64) []float64, hessian func([]float64) [][]float64, x0 []float64) ([]float64, int) {
    x := make([]float64, len(x0))
    copy(x, x0)
    
    for i := 0; i < opt.maxIterations; i++ {
        grad := gradient(x)
        hess := hessian(x)
        
        // Check convergence
        maxGrad := 0.0
        for _, g := range grad {
            if math.Abs(g) > maxGrad {
                maxGrad = math.Abs(g)
            }
        }
        
        if maxGrad < opt.tolerance {
            return x, i
        }
        
        // Solve H * delta = -gradient
        delta := solveLinearSystem(hess, negateVector(grad))
        
        // Update parameters
        for j := 0; j < len(x); j++ {
            x[j] += delta[j]
        }
    }
    
    return x, opt.maxIterations
}

func solveLinearSystem(A [][]float64, b []float64) []float64 {
    n := len(b)
    x := make([]float64, n)
    
    // Simple Gaussian elimination (for small systems)
    // In practice, you'd use more sophisticated methods
    
    // Forward elimination
    for i := 0; i < n; i++ {
        // Find pivot
        maxRow := i
        for k := i + 1; k < n; k++ {
            if math.Abs(A[k][i]) > math.Abs(A[maxRow][i]) {
                maxRow = k
            }
        }
        
        // Swap rows
        A[i], A[maxRow] = A[maxRow], A[i]
        b[i], b[maxRow] = b[maxRow], b[i]
        
        // Make all rows below this one 0 in current column
        for k := i + 1; k < n; k++ {
            if A[i][i] != 0 {
                factor := A[k][i] / A[i][i]
                for j := i; j < n; j++ {
                    A[k][j] -= factor * A[i][j]
                }
                b[k] -= factor * b[i]
            }
        }
    }
    
    // Back substitution
    for i := n - 1; i >= 0; i-- {
        x[i] = b[i]
        for j := i + 1; j < n; j++ {
            x[i] -= A[i][j] * x[j]
        }
        if A[i][i] != 0 {
            x[i] /= A[i][i]
        }
    }
    
    return x
}

func negateVector(v []float64) []float64 {
    result := make([]float64, len(v))
    for i, val := range v {
        result[i] = -val
    }
    return result
}

// Example functions
func rosenbrock(x []float64) float64 {
    if len(x) < 2 {
        return 0
    }
    a := 1.0
    b := 100.0
    return (a-x[0])*(a-x[0]) + b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
}

func rosenbrockGradient(x []float64) []float64 {
    if len(x) < 2 {
        return []float64{0, 0}
    }
    a := 1.0
    b := 100.0
    
    grad := make([]float64, 2)
    grad[0] = -2*(a-x[0]) - 4*b*x[0]*(x[1]-x[0]*x[0])
    grad[1] = 2*b*(x[1]-x[0]*x[0])
    
    return grad
}

func rosenbrockHessian(x []float64) [][]float64 {
    if len(x) < 2 {
        return [][]float64{{0, 0}, {0, 0}}
    }
    a := 1.0
    b := 100.0
    
    hessian := make([][]float64, 2)
    hessian[0] = make([]float64, 2)
    hessian[1] = make([]float64, 2)
    
    hessian[0][0] = 2 + 12*b*x[0]*x[0] - 4*b*x[1]
    hessian[0][1] = -4*b*x[0]
    hessian[1][0] = -4*b*x[0]
    hessian[1][1] = 2*b
    
    return hessian
}

func main() {
    optimizer := NewOptimizer(0.01, 1e-6, 1000)
    
    x0 := []float64{-1.2, 1.0}
    
    fmt.Println("Optimizing Rosenbrock function:")
    fmt.Println("Initial point:", x0)
    fmt.Println("Initial value:", rosenbrock(x0))
    
    xOpt, iterations := optimizer.GradientDescent(rosenbrock, rosenbrockGradient, x0)
    fmt.Printf("Gradient descent result: %v (iterations: %d)\n", xOpt, iterations)
    fmt.Println("Final value:", rosenbrock(xOpt))
    
    xOpt2, iterations2 := optimizer.NewtonMethod(rosenbrock, rosenbrockGradient, rosenbrockHessian, x0)
    fmt.Printf("Newton's method result: %v (iterations: %d)\n", xOpt2, iterations2)
    fmt.Println("Final value:", rosenbrock(xOpt2))
}
```

## Applications

### 1. Machine Learning

#### Linear Regression with Gradient Descent

```go
package main

import (
    "fmt"
    "math"
)

type LinearRegression struct {
    weights []float64
    bias float64
    learningRate float64
}

func NewLinearRegression(learningRate float64) *LinearRegression {
    return &LinearRegression{
        learningRate: learningRate,
    }
}

func (lr *LinearRegression) Fit(X [][]float64, y []float64, epochs int) {
    if len(X) == 0 || len(X[0]) == 0 {
        return
    }
    
    nFeatures := len(X[0])
    nSamples := len(X)
    
    // Initialize weights and bias
    lr.weights = make([]float64, nFeatures)
    lr.bias = 0.0
    
    for epoch := 0; epoch < epochs; epoch++ {
        // Compute predictions
        predictions := make([]float64, nSamples)
        for i := 0; i < nSamples; i++ {
            predictions[i] = lr.predict(X[i])
        }
        
        // Compute gradients
        weightGradients := make([]float64, nFeatures)
        biasGradient := 0.0
        
        for i := 0; i < nSamples; i++ {
            error := predictions[i] - y[i]
            biasGradient += error
            
            for j := 0; j < nFeatures; j++ {
                weightGradients[j] += error * X[i][j]
            }
        }
        
        // Update parameters
        lr.bias -= lr.learningRate * biasGradient / float64(nSamples)
        for j := 0; j < nFeatures; j++ {
            lr.weights[j] -= lr.learningRate * weightGradients[j] / float64(nSamples)
        }
    }
}

func (lr *LinearRegression) predict(x []float64) float64 {
    if len(x) != len(lr.weights) {
        return 0.0
    }
    
    prediction := lr.bias
    for i := 0; i < len(x); i++ {
        prediction += lr.weights[i] * x[i]
    }
    
    return prediction
}

func (lr *LinearRegression) Predict(X [][]float64) []float64 {
    predictions := make([]float64, len(X))
    for i, x := range X {
        predictions[i] = lr.predict(x)
    }
    return predictions
}

func (lr *LinearRegression) Score(X [][]float64, y []float64) float64 {
    predictions := lr.Predict(X)
    
    // Compute R² score
    yMean := 0.0
    for _, yi := range y {
        yMean += yi
    }
    yMean /= float64(len(y))
    
    ssRes := 0.0
    ssTot := 0.0
    
    for i := 0; i < len(y); i++ {
        ssRes += (y[i] - predictions[i]) * (y[i] - predictions[i])
        ssTot += (y[i] - yMean) * (y[i] - yMean)
    }
    
    return 1 - ssRes/ssTot
}

func main() {
    // Example data
    X := [][]float64{
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
    }
    y := []float64{3, 5, 7, 9, 11}
    
    // Create and train model
    model := NewLinearRegression(0.01)
    model.Fit(X, y, 1000)
    
    fmt.Println("Trained weights:", model.weights)
    fmt.Println("Trained bias:", model.bias)
    
    // Make predictions
    testX := [][]float64{{6, 7}, {7, 8}}
    predictions := model.Predict(testX)
    
    fmt.Println("Predictions:", predictions)
    
    // Compute score
    score := model.Score(X, y)
    fmt.Printf("R² Score: %.4f\n", score)
}
```

## Follow-up Questions

### 1. Limits and Continuity
**Q: What is the difference between a limit and a function value?**
A: A limit describes what a function approaches as the input approaches a certain value, while a function value is the actual output at that point. They may be different if the function is discontinuous.

### 2. Derivatives
**Q: Why are derivatives important in optimization?**
A: Derivatives give us the rate of change of a function, which tells us the direction of steepest increase. In optimization, we use this information to find minima or maxima by moving in the opposite direction of the gradient.

### 3. Integration
**Q: When would you use numerical integration instead of analytical integration?**
A: Use numerical integration when the function is too complex for analytical integration, when you only have data points, or when you need a quick approximation for computational purposes.

## Sources

### Books
- **Calculus** by James Stewart
- **Calculus: Early Transcendentals** by Anton, Bivens, Davis
- **Advanced Calculus** by Buck

### Online Resources
- **Khan Academy** - Calculus course
- **3Blue1Brown** - Essence of calculus
- **MIT OpenCourseWare** - Single variable calculus

## Projects

### 1. Function Plotter
**Objective**: Build a mathematical function plotter
**Requirements**: Derivatives, integrals, visualization
**Deliverables**: Interactive plotting application

### 2. Optimization Library
**Objective**: Implement various optimization algorithms
**Requirements**: Gradient descent, Newton's method, line search
**Deliverables**: Complete optimization library

### 3. Numerical Analysis Tool
**Objective**: Create tools for numerical differentiation and integration
**Requirements**: Various numerical methods, error analysis
**Deliverables**: Comprehensive numerical analysis toolkit

---

**Next**: [Statistics & Probability](statistics-probability.md) | **Previous**: [Linear Algebra](linear-algebra.md) | **Up**: [Phase 0](README.md)



## Implementations

<!-- AUTO-GENERATED ANCHOR: originally referenced as #implementations -->

Placeholder content. Please replace with proper section.
