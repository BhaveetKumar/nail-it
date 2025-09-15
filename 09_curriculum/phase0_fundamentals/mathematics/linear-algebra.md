# Linear Algebra for Engineers

## Table of Contents

1. [Overview](#overview)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
5. [Singular Value Decomposition](#singular-value-decomposition)
6. [Applications](#applications)
7. [Implementations](#implementations)
8. [Follow-up Questions](#follow-up-questions)
9. [Sources](#sources)
10. [Projects](#projects)

## Overview

### Learning Objectives

- Master vector operations and properties
- Understand matrix operations and transformations
- Learn eigenvalue decomposition and its applications
- Implement Singular Value Decomposition (SVD)
- Apply linear algebra to computer graphics and machine learning

### What is Linear Algebra?

Linear Algebra is the branch of mathematics that deals with vector spaces, linear transformations, and systems of linear equations. It's fundamental to computer graphics, machine learning, cryptography, and many other areas of computer science.

## Vectors

### 1. Vector Basics

#### Vector Definition
A vector is an ordered collection of numbers (scalars) that can represent direction and magnitude.

```go
package main

import (
    "fmt"
    "math"
)

type Vector struct {
    Components []float64
    Dimension  int
}

func NewVector(components []float64) *Vector {
    return &Vector{
        Components: components,
        Dimension:  len(components),
    }
}

func (v *Vector) Add(other *Vector) *Vector {
    if v.Dimension != other.Dimension {
        panic("Vectors must have the same dimension")
    }
    
    result := make([]float64, v.Dimension)
    for i := 0; i < v.Dimension; i++ {
        result[i] = v.Components[i] + other.Components[i]
    }
    
    return NewVector(result)
}

func (v *Vector) Subtract(other *Vector) *Vector {
    if v.Dimension != other.Dimension {
        panic("Vectors must have the same dimension")
    }
    
    result := make([]float64, v.Dimension)
    for i := 0; i < v.Dimension; i++ {
        result[i] = v.Components[i] - other.Components[i]
    }
    
    return NewVector(result)
}

func (v *Vector) ScalarMultiply(scalar float64) *Vector {
    result := make([]float64, v.Dimension)
    for i := 0; i < v.Dimension; i++ {
        result[i] = v.Components[i] * scalar
    }
    
    return NewVector(result)
}

func (v *Vector) Magnitude() float64 {
    sum := 0.0
    for _, component := range v.Components {
        sum += component * component
    }
    return math.Sqrt(sum)
}

func (v *Vector) Normalize() *Vector {
    magnitude := v.Magnitude()
    if magnitude == 0 {
        panic("Cannot normalize zero vector")
    }
    
    return v.ScalarMultiply(1.0 / magnitude)
}

func (v *Vector) DotProduct(other *Vector) float64 {
    if v.Dimension != other.Dimension {
        panic("Vectors must have the same dimension")
    }
    
    result := 0.0
    for i := 0; i < v.Dimension; i++ {
        result += v.Components[i] * other.Components[i]
    }
    
    return result
}

func (v *Vector) CrossProduct(other *Vector) *Vector {
    if v.Dimension != 3 || other.Dimension != 3 {
        panic("Cross product is only defined for 3D vectors")
    }
    
    result := make([]float64, 3)
    result[0] = v.Components[1]*other.Components[2] - v.Components[2]*other.Components[1]
    result[1] = v.Components[2]*other.Components[0] - v.Components[0]*other.Components[2]
    result[2] = v.Components[0]*other.Components[1] - v.Components[1]*other.Components[0]
    
    return NewVector(result)
}

func (v *Vector) String() string {
    return fmt.Sprintf("Vector%v", v.Components)
}

// Example usage
func main() {
    v1 := NewVector([]float64{1, 2, 3})
    v2 := NewVector([]float64{4, 5, 6})
    
    fmt.Println("Vector 1:", v1)
    fmt.Println("Vector 2:", v2)
    fmt.Println("Sum:", v1.Add(v2))
    fmt.Println("Dot product:", v1.DotProduct(v2))
    fmt.Println("Cross product:", v1.CrossProduct(v2))
    fmt.Println("Magnitude of v1:", v1.Magnitude())
    fmt.Println("Normalized v1:", v1.Normalize())
}
```

#### Node.js Implementation

```javascript
class Vector {
    constructor(components) {
        this.components = components;
        this.dimension = components.length;
    }
    
    add(other) {
        if (this.dimension !== other.dimension) {
            throw new Error('Vectors must have the same dimension');
        }
        
        const result = [];
        for (let i = 0; i < this.dimension; i++) {
            result[i] = this.components[i] + other.components[i];
        }
        
        return new Vector(result);
    }
    
    subtract(other) {
        if (this.dimension !== other.dimension) {
            throw new Error('Vectors must have the same dimension');
        }
        
        const result = [];
        for (let i = 0; i < this.dimension; i++) {
            result[i] = this.components[i] - other.components[i];
        }
        
        return new Vector(result);
    }
    
    scalarMultiply(scalar) {
        const result = [];
        for (let i = 0; i < this.dimension; i++) {
            result[i] = this.components[i] * scalar;
        }
        
        return new Vector(result);
    }
    
    magnitude() {
        let sum = 0;
        for (const component of this.components) {
            sum += component * component;
        }
        return Math.sqrt(sum);
    }
    
    normalize() {
        const magnitude = this.magnitude();
        if (magnitude === 0) {
            throw new Error('Cannot normalize zero vector');
        }
        
        return this.scalarMultiply(1.0 / magnitude);
    }
    
    dotProduct(other) {
        if (this.dimension !== other.dimension) {
            throw new Error('Vectors must have the same dimension');
        }
        
        let result = 0;
        for (let i = 0; i < this.dimension; i++) {
            result += this.components[i] * other.components[i];
        }
        
        return result;
    }
    
    crossProduct(other) {
        if (this.dimension !== 3 || other.dimension !== 3) {
            throw new Error('Cross product is only defined for 3D vectors');
        }
        
        const result = [];
        result[0] = this.components[1] * other.components[2] - 
                   this.components[2] * other.components[1];
        result[1] = this.components[2] * other.components[0] - 
                   this.components[0] * other.components[2];
        result[2] = this.components[0] * other.components[1] - 
                   this.components[1] * other.components[0];
        
        return new Vector(result);
    }
    
    toString() {
        return `Vector[${this.components.join(', ')}]`;
    }
}

// Example usage
const v1 = new Vector([1, 2, 3]);
const v2 = new Vector([4, 5, 6]);

console.log('Vector 1:', v1.toString());
console.log('Vector 2:', v2.toString());
console.log('Sum:', v1.add(v2).toString());
console.log('Dot product:', v1.dotProduct(v2));
console.log('Cross product:', v1.crossProduct(v2).toString());
console.log('Magnitude of v1:', v1.magnitude());
console.log('Normalized v1:', v1.normalize().toString());
```

## Matrices

### 1. Matrix Operations

#### Matrix Definition and Basic Operations

```go
package main

import (
    "fmt"
    "math"
)

type Matrix struct {
    Data [][]float64
    Rows int
    Cols int
}

func NewMatrix(data [][]float64) *Matrix {
    rows := len(data)
    if rows == 0 {
        panic("Matrix cannot be empty")
    }
    cols := len(data[0])
    
    // Validate that all rows have the same number of columns
    for i := 1; i < rows; i++ {
        if len(data[i]) != cols {
            panic("All rows must have the same number of columns")
        }
    }
    
    return &Matrix{
        Data: data,
        Rows: rows,
        Cols: cols,
    }
}

func (m *Matrix) Add(other *Matrix) *Matrix {
    if m.Rows != other.Rows || m.Cols != other.Cols {
        panic("Matrices must have the same dimensions")
    }
    
    result := make([][]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        result[i] = make([]float64, m.Cols)
        for j := 0; j < m.Cols; j++ {
            result[i][j] = m.Data[i][j] + other.Data[i][j]
        }
    }
    
    return NewMatrix(result)
}

func (m *Matrix) Subtract(other *Matrix) *Matrix {
    if m.Rows != other.Rows || m.Cols != other.Cols {
        panic("Matrices must have the same dimensions")
    }
    
    result := make([][]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        result[i] = make([]float64, m.Cols)
        for j := 0; j < m.Cols; j++ {
            result[i][j] = m.Data[i][j] - other.Data[i][j]
        }
    }
    
    return NewMatrix(result)
}

func (m *Matrix) ScalarMultiply(scalar float64) *Matrix {
    result := make([][]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        result[i] = make([]float64, m.Cols)
        for j := 0; j < m.Cols; j++ {
            result[i][j] = m.Data[i][j] * scalar
        }
    }
    
    return NewMatrix(result)
}

func (m *Matrix) Multiply(other *Matrix) *Matrix {
    if m.Cols != other.Rows {
        panic("Number of columns in first matrix must equal number of rows in second matrix")
    }
    
    result := make([][]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        result[i] = make([]float64, other.Cols)
        for j := 0; j < other.Cols; j++ {
            for k := 0; k < m.Cols; k++ {
                result[i][j] += m.Data[i][k] * other.Data[k][j]
            }
        }
    }
    
    return NewMatrix(result)
}

func (m *Matrix) Transpose() *Matrix {
    result := make([][]float64, m.Cols)
    for i := 0; i < m.Cols; i++ {
        result[i] = make([]float64, m.Rows)
        for j := 0; j < m.Rows; j++ {
            result[i][j] = m.Data[j][i]
        }
    }
    
    return NewMatrix(result)
}

func (m *Matrix) Determinant() float64 {
    if m.Rows != m.Cols {
        panic("Determinant is only defined for square matrices")
    }
    
    if m.Rows == 1 {
        return m.Data[0][0]
    }
    
    if m.Rows == 2 {
        return m.Data[0][0]*m.Data[1][1] - m.Data[0][1]*m.Data[1][0]
    }
    
    // For larger matrices, use cofactor expansion
    det := 0.0
    for j := 0; j < m.Cols; j++ {
        cofactor := m.getCofactor(0, j)
        det += m.Data[0][j] * cofactor * math.Pow(-1, float64(j))
    }
    
    return det
}

func (m *Matrix) getCofactor(row, col int) float64 {
    submatrix := m.getSubmatrix(row, col)
    return submatrix.Determinant()
}

func (m *Matrix) getSubmatrix(row, col int) *Matrix {
    result := make([][]float64, m.Rows-1)
    for i := 0; i < m.Rows-1; i++ {
        result[i] = make([]float64, m.Cols-1)
    }
    
    subRow := 0
    for i := 0; i < m.Rows; i++ {
        if i == row {
            continue
        }
        subCol := 0
        for j := 0; j < m.Cols; j++ {
            if j == col {
                continue
            }
            result[subRow][subCol] = m.Data[i][j]
            subCol++
        }
        subRow++
    }
    
    return NewMatrix(result)
}

func (m *Matrix) String() string {
    result := "Matrix[\n"
    for i := 0; i < m.Rows; i++ {
        result += "  ["
        for j := 0; j < m.Cols; j++ {
            result += fmt.Sprintf("%.2f", m.Data[i][j])
            if j < m.Cols-1 {
                result += ", "
            }
        }
        result += "]\n"
    }
    result += "]"
    return result
}

// Example usage
func main() {
    m1 := NewMatrix([][]float64{
        {1, 2},
        {3, 4},
    })
    
    m2 := NewMatrix([][]float64{
        {5, 6},
        {7, 8},
    })
    
    fmt.Println("Matrix 1:")
    fmt.Println(m1)
    fmt.Println("Matrix 2:")
    fmt.Println(m2)
    fmt.Println("Sum:")
    fmt.Println(m1.Add(m2))
    fmt.Println("Product:")
    fmt.Println(m1.Multiply(m2))
    fmt.Println("Transpose of Matrix 1:")
    fmt.Println(m1.Transpose())
    fmt.Println("Determinant of Matrix 1:", m1.Determinant())
}
```

## Eigenvalues and Eigenvectors

### 1. Power Iteration Method

```go
package main

import (
    "fmt"
    "math"
)

func (m *Matrix) PowerIteration(maxIterations int, tolerance float64) (float64, *Vector) {
    if m.Rows != m.Cols {
        panic("Matrix must be square for eigenvalue computation")
    }
    
    // Initialize random vector
    b := make([]float64, m.Rows)
    for i := range b {
        b[i] = 1.0 // Simple initialization
    }
    bVector := NewVector(b)
    bVector = bVector.Normalize()
    
    var eigenvalue float64
    var prevEigenvalue float64
    
    for i := 0; i < maxIterations; i++ {
        // Multiply matrix by vector
        newB := m.MultiplyVector(bVector)
        
        // Find the largest component (approximate eigenvalue)
        eigenvalue = 0
        for j := 0; j < len(newB.Components); j++ {
            if math.Abs(newB.Components[j]) > math.Abs(eigenvalue) {
                eigenvalue = newB.Components[j]
            }
        }
        
        // Normalize the vector
        bVector = newB.Normalize()
        
        // Check for convergence
        if i > 0 && math.Abs(eigenvalue-prevEigenvalue) < tolerance {
            break
        }
        
        prevEigenvalue = eigenvalue
    }
    
    return eigenvalue, bVector
}

func (m *Matrix) MultiplyVector(v *Vector) *Vector {
    if m.Cols != v.Dimension {
        panic("Matrix columns must equal vector dimension")
    }
    
    result := make([]float64, m.Rows)
    for i := 0; i < m.Rows; i++ {
        for j := 0; j < m.Cols; j++ {
            result[i] += m.Data[i][j] * v.Components[j]
        }
    }
    
    return NewVector(result)
}

func (m *Matrix) QRDecomposition() (*Matrix, *Matrix) {
    if m.Rows != m.Cols {
        panic("QR decomposition requires square matrix")
    }
    
    n := m.Rows
    Q := NewMatrix(identityMatrix(n))
    R := NewMatrix(copyMatrix(m.Data))
    
    for j := 0; j < n-1; j++ {
        // Get column j
        col := make([]float64, n-j)
        for i := j; i < n; i++ {
            col[i-j] = R.Data[i][j]
        }
        colVector := NewVector(col)
        
        // Compute Householder vector
        v := colVector.Copy()
        v.Components[0] -= colVector.Magnitude()
        v = v.Normalize()
        
        // Create Householder matrix
        H := householderMatrix(v, j, n)
        
        // Update Q and R
        Q = Q.Multiply(H)
        R = H.Multiply(R)
    }
    
    return Q, R
}

func identityMatrix(n int) [][]float64 {
    result := make([][]float64, n)
    for i := 0; i < n; i++ {
        result[i] = make([]float64, n)
        result[i][i] = 1.0
    }
    return result
}

func copyMatrix(data [][]float64) [][]float64 {
    result := make([][]float64, len(data))
    for i := range data {
        result[i] = make([]float64, len(data[i]))
        copy(result[i], data[i])
    }
    return result
}

func householderMatrix(v *Vector, start, n int) *Matrix {
    H := identityMatrix(n)
    
    for i := start; i < n; i++ {
        for j := start; j < n; j++ {
            if i < len(v.Components) && j < len(v.Components) {
                H[i][j] -= 2 * v.Components[i-start] * v.Components[j-start]
            }
        }
    }
    
    return NewMatrix(H)
}

func (v *Vector) Copy() *Vector {
    components := make([]float64, len(v.Components))
    copy(components, v.Components)
    return NewVector(components)
}
```

## Singular Value Decomposition

### 1. SVD Implementation

```go
package main

import (
    "fmt"
    "math"
)

type SVDResult struct {
    U *Matrix
    S []float64
    V *Matrix
}

func (m *Matrix) SVD() *SVDResult {
    // For simplicity, we'll implement a basic SVD
    // In practice, you'd use more sophisticated algorithms
    
    if m.Rows < m.Cols {
        // Transpose and compute SVD
        transposed := m.Transpose()
        result := transposed.SVD()
        
        // Swap U and V
        return &SVDResult{
            U: result.V,
            S: result.S,
            V: result.U,
        }
    }
    
    // Compute A^T * A
    ATA := m.Transpose().Multiply(m)
    
    // Find eigenvalues and eigenvectors of A^T * A
    eigenvalues, eigenvectors := ATA.PowerIteration(1000, 1e-10)
    
    // Singular values are square roots of eigenvalues
    singularValues := []float64{math.Sqrt(math.Abs(eigenvalues))}
    
    // For simplicity, return basic structure
    // In practice, you'd compute all eigenvalues and eigenvectors
    return &SVDResult{
        U: NewMatrix([][]float64{{1}}), // Placeholder
        S: singularValues,
        V: NewMatrix([][]float64{{1}}), // Placeholder
    }
}

func (svd *SVDResult) Reconstruct() *Matrix {
    // Reconstruct original matrix from SVD components
    // A = U * S * V^T
    
    // This is a simplified reconstruction
    // In practice, you'd properly multiply the matrices
    return NewMatrix([][]float64{{1}}) // Placeholder
}
```

## Applications

### 1. Computer Graphics

#### 3D Transformations

```go
package main

import "math"

type Transform3D struct {
    Matrix *Matrix
}

func NewTransform3D() *Transform3D {
    // Initialize as identity matrix
    identity := [][]float64{
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
    }
    return &Transform3D{Matrix: NewMatrix(identity)}
}

func (t *Transform3D) Translate(x, y, z float64) *Transform3D {
    translation := [][]float64{
        {1, 0, 0, x},
        {0, 1, 0, y},
        {0, 0, 1, z},
        {0, 0, 0, 1},
    }
    
    t.Matrix = t.Matrix.Multiply(NewMatrix(translation))
    return t
}

func (t *Transform3D) RotateX(angle float64) *Transform3D {
    cos := math.Cos(angle)
    sin := math.Sin(angle)
    
    rotation := [][]float64{
        {1, 0, 0, 0},
        {0, cos, -sin, 0},
        {0, sin, cos, 0},
        {0, 0, 0, 1},
    }
    
    t.Matrix = t.Matrix.Multiply(NewMatrix(rotation))
    return t
}

func (t *Transform3D) RotateY(angle float64) *Transform3D {
    cos := math.Cos(angle)
    sin := math.Sin(angle)
    
    rotation := [][]float64{
        {cos, 0, sin, 0},
        {0, 1, 0, 0},
        {-sin, 0, cos, 0},
        {0, 0, 0, 1},
    }
    
    t.Matrix = t.Matrix.Multiply(NewMatrix(rotation))
    return t
}

func (t *Transform3D) RotateZ(angle float64) *Transform3D {
    cos := math.Cos(angle)
    sin := math.Sin(angle)
    
    rotation := [][]float64{
        {cos, -sin, 0, 0},
        {sin, cos, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
    }
    
    t.Matrix = t.Matrix.Multiply(NewMatrix(rotation))
    return t
}

func (t *Transform3D) Scale(x, y, z float64) *Transform3D {
    scaling := [][]float64{
        {x, 0, 0, 0},
        {0, y, 0, 0},
        {0, 0, z, 0},
        {0, 0, 0, 1},
    }
    
    t.Matrix = t.Matrix.Multiply(NewMatrix(scaling))
    return t
}

func (t *Transform3D) TransformPoint(x, y, z float64) (float64, float64, float64) {
    point := NewVector([]float64{x, y, z, 1})
    transformed := t.Matrix.MultiplyVector(point)
    
    // Convert back from homogeneous coordinates
    w := transformed.Components[3]
    return transformed.Components[0]/w, transformed.Components[1]/w, transformed.Components[2]/w
}
```

### 2. Machine Learning

#### Principal Component Analysis

```go
package main

import "math"

type PCA struct {
    Components *Matrix
    ExplainedVariance []float64
    Mean *Vector
}

func NewPCA(data [][]float64) *PCA {
    // Center the data
    centered := centerData(data)
    
    // Compute covariance matrix
    cov := computeCovarianceMatrix(centered)
    
    // Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors := cov.PowerIteration(1000, 1e-10)
    
    // Sort by eigenvalue (simplified)
    components := eigenvectors
    explainedVariance := []float64{math.Abs(eigenvalues)}
    
    return &PCA{
        Components: NewMatrix([][]float64{eigenvectors.Components}),
        ExplainedVariance: explainedVariance,
        Mean: computeMean(data),
    }
}

func centerData(data [][]float64) [][]float64 {
    if len(data) == 0 {
        return data
    }
    
    n := len(data)
    d := len(data[0])
    mean := computeMean(data)
    
    centered := make([][]float64, n)
    for i := 0; i < n; i++ {
        centered[i] = make([]float64, d)
        for j := 0; j < d; j++ {
            centered[i][j] = data[i][j] - mean.Components[j]
        }
    }
    
    return centered
}

func computeMean(data [][]float64) *Vector {
    if len(data) == 0 {
        return NewVector([]float64{})
    }
    
    n := len(data)
    d := len(data[0])
    mean := make([]float64, d)
    
    for i := 0; i < n; i++ {
        for j := 0; j < d; j++ {
            mean[j] += data[i][j]
        }
    }
    
    for j := 0; j < d; j++ {
        mean[j] /= float64(n)
    }
    
    return NewVector(mean)
}

func computeCovarianceMatrix(data [][]float64) *Matrix {
    n := len(data)
    d := len(data[0])
    
    cov := make([][]float64, d)
    for i := 0; i < d; i++ {
        cov[i] = make([]float64, d)
    }
    
    for i := 0; i < d; i++ {
        for j := 0; j < d; j++ {
            sum := 0.0
            for k := 0; k < n; k++ {
                sum += data[k][i] * data[k][j]
            }
            cov[i][j] = sum / float64(n-1)
        }
    }
    
    return NewMatrix(cov)
}

func (pca *PCA) Transform(data [][]float64, nComponents int) [][]float64 {
    // Center the data
    centered := centerData(data)
    
    // Project onto principal components
    result := make([][]float64, len(centered))
    for i := 0; i < len(centered); i++ {
        result[i] = make([]float64, nComponents)
        for j := 0; j < nComponents; j++ {
            sum := 0.0
            for k := 0; k < len(centered[i]); k++ {
                sum += centered[i][k] * pca.Components.Data[j][k]
            }
            result[i][j] = sum
        }
    }
    
    return result
}
```

## Follow-up Questions

### 1. Vector Operations
**Q: What is the geometric interpretation of the dot product?**
A: The dot product represents the projection of one vector onto another, scaled by the magnitude of the second vector. It's also related to the angle between vectors: a·b = |a||b|cos(θ).

### 2. Matrix Properties
**Q: What makes a matrix invertible?**
A: A matrix is invertible (non-singular) if and only if its determinant is non-zero. This means the matrix has full rank and its columns are linearly independent.

### 3. Eigenvalues
**Q: Why are eigenvalues important in machine learning?**
A: Eigenvalues help identify the principal directions of variation in data (PCA), determine system stability, and are used in dimensionality reduction techniques.

## Sources

### Books
- **Linear Algebra Done Right** by Sheldon Axler
- **Introduction to Linear Algebra** by Gilbert Strang
- **Linear Algebra and Its Applications** by David Lay

### Online Resources
- **Khan Academy** - Linear algebra course
- **3Blue1Brown** - Essence of linear algebra
- **MIT OpenCourseWare** - Linear algebra course

## Projects

### 1. 3D Graphics Engine
**Objective**: Build a simple 3D graphics engine
**Requirements**: Matrix transformations, vector operations
**Deliverables**: Working 3D renderer with rotation and scaling

### 2. PCA Implementation
**Objective**: Implement Principal Component Analysis
**Requirements**: Eigenvalue computation, data transformation
**Deliverables**: Complete PCA library with visualization

### 3. Linear System Solver
**Objective**: Solve systems of linear equations
**Requirements**: Gaussian elimination, matrix operations
**Deliverables**: Robust linear system solver

---

**Next**: [Calculus](./calculus.md) | **Previous**: [Mathematics](../README.md) | **Up**: [Phase 0](../README.md)
