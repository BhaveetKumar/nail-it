# Rotate Image

### Problem
You are given an `n x n` 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

**Example:**
```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
```

**Constraints:**
- n == matrix.length == matrix[i].length
- 1 ≤ n ≤ 20
- -1000 ≤ matrix[i][j] ≤ 1000

### Explanation

#### **Transpose + Reverse Approach**
- First transpose the matrix (swap matrix[i][j] with matrix[j][i])
- Then reverse each row
- Time Complexity: O(n²)
- Space Complexity: O(1)

### Golang Solution

```go
func rotate(matrix [][]int) {
    n := len(matrix)
    
    // Transpose the matrix
    for i := 0; i < n; i++ {
        for j := i; j < n; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
    
    // Reverse each row
    for i := 0; i < n; i++ {
        for j := 0; j < n/2; j++ {
            matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
        }
    }
}
```

### Notes / Variations

#### **Related Problems**
- **Spiral Matrix**: Traverse matrix in spiral order
- **Set Matrix Zeroes**: Set entire row/column to zero
- **Valid Sudoku**: Validate Sudoku board
- **Game of Life**: Simulate cellular automaton
