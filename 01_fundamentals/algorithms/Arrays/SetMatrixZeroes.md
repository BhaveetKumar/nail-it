# Set Matrix Zeroes

### Problem
Given an `m x n` integer matrix `matrix`, if an element is `0`, set its entire row and column to `0`s.

You must do it in-place.

**Example:**
```
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
```

### Golang Solution

```go
func setZeroes(matrix [][]int) {
    m, n := len(matrix), len(matrix[0])
    firstRowZero := false
    firstColZero := false
    
    // Check if first row has zero
    for j := 0; j < n; j++ {
        if matrix[0][j] == 0 {
            firstRowZero = true
            break
        }
    }
    
    // Check if first column has zero
    for i := 0; i < m; i++ {
        if matrix[i][0] == 0 {
            firstColZero = true
            break
        }
    }
    
    // Use first row and column as markers
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if matrix[i][j] == 0 {
                matrix[i][0] = 0
                matrix[0][j] = 0
            }
        }
    }
    
    // Set zeros based on markers
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if matrix[i][0] == 0 || matrix[0][j] == 0 {
                matrix[i][j] = 0
            }
        }
    }
    
    // Handle first row
    if firstRowZero {
        for j := 0; j < n; j++ {
            matrix[0][j] = 0
        }
    }
    
    // Handle first column
    if firstColZero {
        for i := 0; i < m; i++ {
            matrix[i][0] = 0
        }
    }
}
```

### Alternative Solutions

#### **Using Extra Space**
```go
func setZeroesExtraSpace(matrix [][]int) {
    m, n := len(matrix), len(matrix[0])
    zeroRows := make([]bool, m)
    zeroCols := make([]bool, n)
    
    // Find zeros
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if matrix[i][j] == 0 {
                zeroRows[i] = true
                zeroCols[j] = true
            }
        }
    }
    
    // Set zeros
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if zeroRows[i] || zeroCols[j] {
                matrix[i][j] = 0
            }
        }
    }
}
```

#### **Using Sets**
```go
func setZeroesSets(matrix [][]int) {
    m, n := len(matrix), len(matrix[0])
    zeroRows := make(map[int]bool)
    zeroCols := make(map[int]bool)
    
    // Find zeros
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if matrix[i][j] == 0 {
                zeroRows[i] = true
                zeroCols[j] = true
            }
        }
    }
    
    // Set zeros
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if zeroRows[i] || zeroCols[j] {
                matrix[i][j] = 0
            }
        }
    }
}
```

#### **Two Pass Approach**
```go
func setZeroesTwoPass(matrix [][]int) {
    m, n := len(matrix), len(matrix[0])
    
    // First pass: mark rows and columns to be zeroed
    rowsToZero := make([]bool, m)
    colsToZero := make([]bool, n)
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if matrix[i][j] == 0 {
                rowsToZero[i] = true
                colsToZero[j] = true
            }
        }
    }
    
    // Second pass: set zeros
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if rowsToZero[i] || colsToZero[j] {
                matrix[i][j] = 0
            }
        }
    }
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1) for in-place, O(m + n) for extra space