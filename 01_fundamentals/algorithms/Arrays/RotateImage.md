---
# Auto-generated front matter
Title: Rotateimage
LastUpdated: 2025-11-06T20:45:58.724525
Tags: []
Status: draft
---

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

### Alternative Solutions

#### **Layer by Layer Rotation**

```go
func rotateLayerByLayer(matrix [][]int) {
    n := len(matrix)

    for layer := 0; layer < n/2; layer++ {
        first := layer
        last := n - 1 - layer

        for i := first; i < last; i++ {
            offset := i - first

            // Save top
            top := matrix[first][i]

            // Move left to top
            matrix[first][i] = matrix[last-offset][first]

            // Move bottom to left
            matrix[last-offset][first] = matrix[last][last-offset]

            // Move right to bottom
            matrix[last][last-offset] = matrix[i][last]

            // Move top to right
            matrix[i][last] = top
        }
    }
}
```

#### **Using Extra Space**

```go
func rotateExtraSpace(matrix [][]int) {
    n := len(matrix)
    rotated := make([][]int, n)
    for i := range rotated {
        rotated[i] = make([]int, n)
    }

    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            rotated[j][n-1-i] = matrix[i][j]
        }
    }

    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            matrix[i][j] = rotated[i][j]
        }
    }
}
```

### Complexity

- **Time Complexity:** O(n²)
- **Space Complexity:** O(1) for in-place, O(n²) for extra space
