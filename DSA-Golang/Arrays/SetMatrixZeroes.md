# Set Matrix Zeroes

### Problem
Given an `m x n` integer matrix `matrix`, if an element is `0`, set its entire row and column to `0`s.

You must do it in-place.

**Example:**
```
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
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
    
    // Set first row
    if firstRowZero {
        for j := 0; j < n; j++ {
            matrix[0][j] = 0
        }
    }
    
    // Set first column
    if firstColZero {
        for i := 0; i < m; i++ {
            matrix[i][0] = 0
        }
    }
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1)
