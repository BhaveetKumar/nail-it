# Advanced Numerical Algorithms

## Table of Contents
- [Introduction](#introduction/)
- [Root Finding](#root-finding/)
- [Numerical Integration](#numerical-integration/)
- [Linear Algebra](#linear-algebra/)
- [Optimization](#optimization/)
- [Differential Equations](#differential-equations/)
- [Advanced Applications](#advanced-applications/)

## Introduction

Advanced numerical algorithms provide sophisticated methods for solving complex mathematical problems with high precision and efficiency.

## Root Finding

### Newton-Raphson Method

**Problem**: Find roots of nonlinear equations with quadratic convergence.

```go
// Newton-Raphson Method
type NewtonRaphson struct {
    function    func(float64) float64
    derivative  func(float64) float64
    tolerance   float64
    maxIterations int
}

func NewNewtonRaphson(f, df func(float64) float64) *NewtonRaphson {
    return &NewtonRaphson{
        function:      f,
        derivative:    df,
        tolerance:     1e-10,
        maxIterations: 100,
    }
}

func (nr *NewtonRaphson) FindRoot(initialGuess float64) (float64, error) {
    x := initialGuess
    
    for i := 0; i < nr.maxIterations; i++ {
        fx := nr.function(x)
        
        if math.Abs(fx) < nr.tolerance {
            return x, nil
        }
        
        dfx := nr.derivative(x)
        if math.Abs(dfx) < 1e-15 {
            return 0, fmt.Errorf("derivative is zero at x = %f", x)
        }
        
        xNew := x - fx/dfx
        
        if math.Abs(xNew-x) < nr.tolerance {
            return xNew, nil
        }
        
        x = xNew
    }
    
    return 0, fmt.Errorf("convergence not achieved after %d iterations", nr.maxIterations)
}

// Secant Method
type SecantMethod struct {
    function    func(float64) float64
    tolerance   float64
    maxIterations int
}

func NewSecantMethod(f func(float64) float64) *SecantMethod {
    return &SecantMethod{
        function:      f,
        tolerance:     1e-10,
        maxIterations: 100,
    }
}

func (sm *SecantMethod) FindRoot(x0, x1 float64) (float64, error) {
    for i := 0; i < sm.maxIterations; i++ {
        fx0 := sm.function(x0)
        fx1 := sm.function(x1)
        
        if math.Abs(fx1) < sm.tolerance {
            return x1, nil
        }
        
        if math.Abs(fx1-fx0) < 1e-15 {
            return 0, fmt.Errorf("function values are too close")
        }
        
        x2 := x1 - fx1*(x1-x0)/(fx1-fx0)
        
        if math.Abs(x2-x1) < sm.tolerance {
            return x2, nil
        }
        
        x0, x1 = x1, x2
    }
    
    return 0, fmt.Errorf("convergence not achieved after %d iterations", sm.maxIterations)
}
```

## Numerical Integration

### Gaussian Quadrature

**Problem**: Approximate definite integrals with high accuracy.

```go
// Gaussian Quadrature
type GaussianQuadrature struct {
    nodes []float64
    weights []float64
    degree int
}

func NewGaussianQuadrature(degree int) *GaussianQuadrature {
    gq := &GaussianQuadrature{
        degree: degree,
    }
    
    gq.computeNodesAndWeights()
    return gq
}

func (gq *GaussianQuadrature) Integrate(f func(float64) float64, a, b float64) float64 {
    // Transform interval [a,b] to [-1,1]
    sum := 0.0
    
    for i := 0; i < gq.degree; i++ {
        x := gq.transformToInterval(gq.nodes[i], a, b)
        sum += gq.weights[i] * f(x)
    }
    
    // Scale by Jacobian
    return sum * (b - a) / 2.0
}

func (gq *GaussianQuadrature) transformToInterval(x, a, b float64) float64 {
    return (b-a)*x/2.0 + (a+b)/2.0
}

func (gq *GaussianQuadrature) computeNodesAndWeights() {
    // Simplified implementation for common degrees
    switch gq.degree {
    case 2:
        gq.nodes = []float64{-0.5773502691896257, 0.5773502691896257}
        gq.weights = []float64{1.0, 1.0}
    case 3:
        gq.nodes = []float64{-0.7745966692414834, 0.0, 0.7745966692414834}
        gq.weights = []float64{0.5555555555555556, 0.8888888888888888, 0.5555555555555556}
    case 4:
        gq.nodes = []float64{-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526}
        gq.weights = []float64{0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538}
    default:
        // Use general method for higher degrees
        gq.computeGeneralNodesAndWeights()
    }
}

func (gq *GaussianQuadrature) computeGeneralNodesAndWeights() {
    // Simplified general computation
    // In practice, this would use more sophisticated methods
    gq.nodes = make([]float64, gq.degree)
    gq.weights = make([]float64, gq.degree)
    
    for i := 0; i < gq.degree; i++ {
        gq.nodes[i] = math.Cos(math.Pi * float64(2*i+1) / float64(2*gq.degree))
        gq.weights[i] = math.Pi / float64(gq.degree)
    }
}

// Adaptive Quadrature
type AdaptiveQuadrature struct {
    tolerance float64
    maxDepth  int
}

func NewAdaptiveQuadrature() *AdaptiveQuadrature {
    return &AdaptiveQuadrature{
        tolerance: 1e-10,
        maxDepth:  20,
    }
}

func (aq *AdaptiveQuadrature) Integrate(f func(float64) float64, a, b float64) float64 {
    return aq.adaptiveIntegrate(f, a, b, 0)
}

func (aq *AdaptiveQuadrature) adaptiveIntegrate(f func(float64) float64, a, b float64, depth int) float64 {
    if depth > aq.maxDepth {
        return aq.simpsonRule(f, a, b)
    }
    
    mid := (a + b) / 2.0
    
    // Compute integrals
    integral1 := aq.simpsonRule(f, a, b)
    integral2 := aq.simpsonRule(f, a, mid) + aq.simpsonRule(f, mid, b)
    
    // Check error
    error := math.Abs(integral1 - integral2)
    
    if error < aq.tolerance {
        return integral2
    }
    
    // Recursively subdivide
    return aq.adaptiveIntegrate(f, a, mid, depth+1) + aq.adaptiveIntegrate(f, mid, b, depth+1)
}

func (aq *AdaptiveQuadrature) simpsonRule(f func(float64) float64, a, b float64) float64 {
    h := (b - a) / 2.0
    return h / 3.0 * (f(a) + 4*f(a+h) + f(b))
}
```

## Linear Algebra

### LU Decomposition

**Problem**: Solve linear systems using LU decomposition.

```go
// LU Decomposition
type LUDecomposition struct {
    matrix [][]float64
    L      [][]float64
    U      [][]float64
    P      []int
    n      int
}

func NewLUDecomposition(matrix [][]float64) *LUDecomposition {
    n := len(matrix)
    lu := &LUDecomposition{
        matrix: matrix,
        L:      make([][]float64, n),
        U:      make([][]float64, n),
        P:      make([]int, n),
        n:      n,
    }
    
    for i := 0; i < n; i++ {
        lu.L[i] = make([]float64, n)
        lu.U[i] = make([]float64, n)
        lu.P[i] = i
    }
    
    lu.decompose()
    return lu
}

func (lu *LUDecomposition) decompose() {
    // Copy matrix to U
    for i := 0; i < lu.n; i++ {
        for j := 0; j < lu.n; j++ {
            lu.U[i][j] = lu.matrix[i][j]
        }
    }
    
    // Initialize L as identity matrix
    for i := 0; i < lu.n; i++ {
        lu.L[i][i] = 1.0
    }
    
    // Perform Gaussian elimination with partial pivoting
    for k := 0; k < lu.n-1; k++ {
        // Find pivot
        maxRow := k
        for i := k + 1; i < lu.n; i++ {
            if math.Abs(lu.U[i][k]) > math.Abs(lu.U[maxRow][k]) {
                maxRow = i
            }
        }
        
        // Swap rows
        if maxRow != k {
            lu.swapRows(k, maxRow)
            lu.P[k], lu.P[maxRow] = lu.P[maxRow], lu.P[k]
        }
        
        // Check for singular matrix
        if math.Abs(lu.U[k][k]) < 1e-15 {
            panic("Matrix is singular")
        }
        
        // Eliminate
        for i := k + 1; i < lu.n; i++ {
            factor := lu.U[i][k] / lu.U[k][k]
            lu.L[i][k] = factor
            
            for j := k; j < lu.n; j++ {
                lu.U[i][j] -= factor * lu.U[k][j]
            }
        }
    }
}

func (lu *LUDecomposition) swapRows(i, j int) {
    lu.U[i], lu.U[j] = lu.U[j], lu.U[i]
}

func (lu *LUDecomposition) Solve(b []float64) []float64 {
    // Solve Ly = Pb
    y := make([]float64, lu.n)
    pb := make([]float64, lu.n)
    
    for i := 0; i < lu.n; i++ {
        pb[i] = b[lu.P[i]]
    }
    
    for i := 0; i < lu.n; i++ {
        y[i] = pb[i]
        for j := 0; j < i; j++ {
            y[i] -= lu.L[i][j] * y[j]
        }
    }
    
    // Solve Ux = y
    x := make([]float64, lu.n)
    for i := lu.n - 1; i >= 0; i-- {
        x[i] = y[i]
        for j := i + 1; j < lu.n; j++ {
            x[i] -= lu.U[i][j] * x[j]
        }
        x[i] /= lu.U[i][i]
    }
    
    return x
}

// QR Decomposition
type QRDecomposition struct {
    Q [][]float64
    R [][]float64
    n int
}

func NewQRDecomposition(matrix [][]float64) *QRDecomposition {
    n := len(matrix)
    qr := &QRDecomposition{
        Q: make([][]float64, n),
        R: make([][]float64, n),
        n: n,
    }
    
    for i := 0; i < n; i++ {
        qr.Q[i] = make([]float64, n)
        qr.R[i] = make([]float64, n)
    }
    
    qr.decompose(matrix)
    return qr
}

func (qr *QRDecomposition) decompose(matrix [][]float64) {
    // Copy matrix to R
    for i := 0; i < qr.n; i++ {
        for j := 0; j < qr.n; j++ {
            qr.R[i][j] = matrix[i][j]
        }
    }
    
    // Initialize Q as identity matrix
    for i := 0; i < qr.n; i++ {
        qr.Q[i][i] = 1.0
    }
    
    // Gram-Schmidt process
    for k := 0; k < qr.n; k++ {
        // Normalize k-th column
        norm := 0.0
        for i := 0; i < qr.n; i++ {
            norm += qr.R[i][k] * qr.R[i][k]
        }
        norm = math.Sqrt(norm)
        
        if norm < 1e-15 {
            panic("Matrix is singular")
        }
        
        qr.R[k][k] = norm
        for i := 0; i < qr.n; i++ {
            qr.Q[i][k] = qr.R[i][k] / norm
        }
        
        // Orthogonalize remaining columns
        for j := k + 1; j < qr.n; j++ {
            dot := 0.0
            for i := 0; i < qr.n; i++ {
                dot += qr.Q[i][k] * qr.R[i][j]
            }
            
            qr.R[k][j] = dot
            
            for i := 0; i < qr.n; i++ {
                qr.R[i][j] -= dot * qr.Q[i][k]
            }
        }
    }
}
```

## Optimization

### Gradient Descent

**Problem**: Minimize functions using gradient descent.

```go
// Gradient Descent
type GradientDescent struct {
    learningRate float64
    tolerance    float64
    maxIterations int
}

func NewGradientDescent() *GradientDescent {
    return &GradientDescent{
        learningRate:   0.01,
        tolerance:      1e-8,
        maxIterations:  1000,
    }
}

func (gd *GradientDescent) Minimize(f func([]float64) float64, gradient func([]float64) []float64, initial []float64) ([]float64, error) {
    x := make([]float64, len(initial))
    copy(x, initial)
    
    for i := 0; i < gd.maxIterations; i++ {
        grad := gradient(x)
        
        // Check convergence
        norm := 0.0
        for _, g := range grad {
            norm += g * g
        }
        norm = math.Sqrt(norm)
        
        if norm < gd.tolerance {
            return x, nil
        }
        
        // Update parameters
        for j := 0; j < len(x); j++ {
            x[j] -= gd.learningRate * grad[j]
        }
    }
    
    return x, fmt.Errorf("convergence not achieved after %d iterations", gd.maxIterations)
}

// Conjugate Gradient Method
type ConjugateGradient struct {
    tolerance    float64
    maxIterations int
}

func NewConjugateGradient() *ConjugateGradient {
    return &ConjugateGradient{
        tolerance:      1e-10,
        maxIterations:  1000,
    }
}

func (cg *ConjugateGradient) Solve(A [][]float64, b []float64, initial []float64) ([]float64, error) {
    n := len(b)
    x := make([]float64, n)
    copy(x, initial)
    
    // Compute initial residual
    r := make([]float64, n)
    for i := 0; i < n; i++ {
        r[i] = b[i]
        for j := 0; j < n; j++ {
            r[i] -= A[i][j] * x[j]
        }
    }
    
    p := make([]float64, n)
    copy(p, r)
    
    for k := 0; k < cg.maxIterations; k++ {
        // Compute Ap
        Ap := make([]float64, n)
        for i := 0; i < n; i++ {
            for j := 0; j < n; j++ {
                Ap[i] += A[i][j] * p[j]
            }
        }
        
        // Compute alpha
        rDotR := 0.0
        pDotAp := 0.0
        for i := 0; i < n; i++ {
            rDotR += r[i] * r[i]
            pDotAp += p[i] * Ap[i]
        }
        
        if math.Abs(pDotAp) < 1e-15 {
            return x, nil
        }
        
        alpha := rDotR / pDotAp
        
        // Update solution and residual
        for i := 0; i < n; i++ {
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        }
        
        // Check convergence
        rNorm := 0.0
        for i := 0; i < n; i++ {
            rNorm += r[i] * r[i]
        }
        rNorm = math.Sqrt(rNorm)
        
        if rNorm < cg.tolerance {
            return x, nil
        }
        
        // Compute beta
        rNewDotRNew := 0.0
        for i := 0; i < n; i++ {
            rNewDotRNew += r[i] * r[i]
        }
        
        beta := rNewDotRNew / rDotR
        
        // Update search direction
        for i := 0; i < n; i++ {
            p[i] = r[i] + beta*p[i]
        }
    }
    
    return x, fmt.Errorf("convergence not achieved after %d iterations", cg.maxIterations)
}
```

## Differential Equations

### Runge-Kutta Methods

**Problem**: Solve ordinary differential equations numerically.

```go
// Runge-Kutta 4th Order
type RungeKutta4 struct {
    stepSize float64
    tolerance float64
}

func NewRungeKutta4(stepSize float64) *RungeKutta4 {
    return &RungeKutta4{
        stepSize:  stepSize,
        tolerance: 1e-8,
    }
}

func (rk4 *RungeKutta4) Solve(f func(float64, float64) float64, y0 float64, t0, tEnd float64) []Point {
    var solution []Point
    t := t0
    y := y0
    
    solution = append(solution, Point{X: t, Y: y})
    
    for t < tEnd {
        // RK4 step
        k1 := rk4.stepSize * f(t, y)
        k2 := rk4.stepSize * f(t+rk4.stepSize/2, y+k1/2)
        k3 := rk4.stepSize * f(t+rk4.stepSize/2, y+k2/2)
        k4 := rk4.stepSize * f(t+rk4.stepSize, y+k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += rk4.stepSize
        
        solution = append(solution, Point{X: t, Y: y})
    }
    
    return solution
}

type Point struct {
    X, Y float64
}

// Adaptive Runge-Kutta
type AdaptiveRungeKutta struct {
    minStepSize float64
    maxStepSize float64
    tolerance   float64
}

func NewAdaptiveRungeKutta() *AdaptiveRungeKutta {
    return &AdaptiveRungeKutta{
        minStepSize: 1e-8,
        maxStepSize: 0.1,
        tolerance:   1e-8,
    }
}

func (ark *AdaptiveRungeKutta) Solve(f func(float64, float64) float64, y0 float64, t0, tEnd float64) []Point {
    var solution []Point
    t := t0
    y := y0
    h := ark.maxStepSize
    
    solution = append(solution, Point{X: t, Y: y})
    
    for t < tEnd {
        // Take two half steps
        y1 := ark.rk4Step(f, t, y, h/2)
        y2 := ark.rk4Step(f, t+h/2, y1, h/2)
        
        // Take one full step
        yFull := ark.rk4Step(f, t, y, h)
        
        // Estimate error
        error := math.Abs(y2 - yFull)
        
        if error < ark.tolerance {
            // Accept step
            y = y2
            t += h
            solution = append(solution, Point{X: t, Y: y})
            
            // Increase step size
            h = math.Min(ark.maxStepSize, h*1.2)
        } else {
            // Reject step and decrease step size
            h = math.Max(ark.minStepSize, h*0.5)
        }
    }
    
    return solution
}

func (ark *AdaptiveRungeKutta) rk4Step(f func(float64, float64) float64, t, y, h float64) float64 {
    k1 := h * f(t, y)
    k2 := h * f(t+h/2, y+k1/2)
    k3 := h * f(t+h/2, y+k2/2)
    k4 := h * f(t+h, y+k3)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6
}
```

## Conclusion

Advanced numerical algorithms provide:

1. **Precision**: High-accuracy numerical methods
2. **Efficiency**: Optimized algorithms for large problems
3. **Stability**: Robust numerical techniques
4. **Convergence**: Guaranteed convergence properties
5. **Applications**: Real-world mathematical problems
6. **Optimization**: Advanced optimization techniques
7. **Analysis**: Numerical analysis and error estimation

Mastering these algorithms prepares you for complex mathematical problems in technical interviews and real-world applications.

## Additional Resources

- [Numerical Analysis](https://www.numericalanalysis.com/)
- [Root Finding](https://www.rootfinding.com/)
- [Numerical Integration](https://www.numericalintegration.com/)
- [Linear Algebra](https://www.linearalgebra.com/)
- [Optimization](https://www.optimization.com/)
- [Differential Equations](https://www.differentialequations.com/)
- [Numerical Methods](https://www.numericalmethods.com/)
- [Mathematical Computing](https://www.mathematicalcomputing.com/)
