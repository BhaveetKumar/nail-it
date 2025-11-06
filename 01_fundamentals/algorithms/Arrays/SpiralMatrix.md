---
# Auto-generated front matter
Title: Spiralmatrix
LastUpdated: 2025-11-06T20:45:58.721165
Tags: []
Status: draft
---

# Spiral Matrix

### Problem

Given an `m x n` matrix, return all elements of the matrix in spiral order.

**Example:**

```
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```

### Golang Solution

```go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }

    m, n := len(matrix), len(matrix[0])
    result := make([]int, 0, m*n)

    top, bottom := 0, m-1
    left, right := 0, n-1

    for top <= bottom && left <= right {
        // Traverse right
        for j := left; j <= right; j++ {
            result = append(result, matrix[top][j])
        }
        top++

        // Traverse down
        for i := top; i <= bottom; i++ {
            result = append(result, matrix[i][right])
        }
        right--

        // Traverse left
        if top <= bottom {
            for j := right; j >= left; j-- {
                result = append(result, matrix[bottom][j])
            }
            bottom--
        }

        // Traverse up
        if left <= right {
            for i := bottom; i >= top; i-- {
                result = append(result, matrix[i][left])
            }
            left++
        }
    }

    return result
}
```

### Alternative Solutions

#### **Direction-based Approach**

```go
func spiralOrderDirection(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }

    m, n := len(matrix), len(matrix[0])
    result := make([]int, 0, m*n)

    directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
    visited := make([][]bool, m)
    for i := range visited {
        visited[i] = make([]bool, n)
    }

    row, col, dir := 0, 0, 0

    for i := 0; i < m*n; i++ {
        result = append(result, matrix[row][col])
        visited[row][col] = true

        nextRow := row + directions[dir][0]
        nextCol := col + directions[dir][1]

        if nextRow < 0 || nextRow >= m || nextCol < 0 || nextCol >= n || visited[nextRow][nextCol] {
            dir = (dir + 1) % 4
            nextRow = row + directions[dir][0]
            nextCol = col + directions[dir][1]
        }

        row, col = nextRow, nextCol
    }

    return result
}
```

#### **Recursive Approach**

```go
func spiralOrderRecursive(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }

    var result []int
    spiralHelper(matrix, 0, 0, len(matrix)-1, len(matrix[0])-1, &result)
    return result
}

func spiralHelper(matrix [][]int, top, left, bottom, right int, result *[]int) {
    if top > bottom || left > right {
        return
    }

    // Traverse right
    for j := left; j <= right; j++ {
        *result = append(*result, matrix[top][j])
    }

    // Traverse down
    for i := top + 1; i <= bottom; i++ {
        *result = append(*result, matrix[i][right])
    }

    // Traverse left
    if top < bottom {
        for j := right - 1; j >= left; j-- {
            *result = append(*result, matrix[bottom][j])
        }
    }

    // Traverse up
    if left < right {
        for i := bottom - 1; i > top; i-- {
            *result = append(*result, matrix[i][left])
        }
    }

    spiralHelper(matrix, top+1, left+1, bottom-1, right-1, result)
}
```

### Complexity

- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1) for iterative, O(m × n) for direction-based
