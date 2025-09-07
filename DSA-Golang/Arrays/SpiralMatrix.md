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

**Constraints:**
- m == matrix.length
- n == matrix[i].length
- 1 ≤ m, n ≤ 10
- -100 ≤ matrix[i][j] ≤ 100

### Explanation

#### **Boundary Approach**
- Define four boundaries: top, bottom, left, right
- Traverse in four directions: right, down, left, up
- Shrink boundaries after each direction
- Time Complexity: O(m × n)
- Space Complexity: O(1)

### Dry Run

**Input:** `matrix = [[1,2,3],[4,5,6],[7,8,9]]`

| Step | Direction | Boundaries | Elements | Action |
|------|-----------|------------|----------|---------|
| 1 | Right | top=0, bottom=2, left=0, right=2 | 1,2,3 | top++ |
| 2 | Down | top=1, bottom=2, left=0, right=2 | 6,9 | right-- |
| 3 | Left | top=1, bottom=2, left=0, right=1 | 8,7 | bottom-- |
| 4 | Up | top=1, bottom=1, left=0, right=1 | 4 | left++ |

**Result:** `[1,2,3,6,9,8,7,4,5]`

### Complexity
- **Time Complexity:** O(m × n) - Visit each element once
- **Space Complexity:** O(1) - Only using constant extra space

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
        for col := left; col <= right; col++ {
            result = append(result, matrix[top][col])
        }
        top++
        
        // Traverse down
        for row := top; row <= bottom; row++ {
            result = append(result, matrix[row][right])
        }
        right--
        
        // Traverse left (if there are rows left)
        if top <= bottom {
            for col := right; col >= left; col-- {
                result = append(result, matrix[bottom][col])
            }
            bottom--
        }
        
        // Traverse up (if there are columns left)
        if left <= right {
            for row := bottom; row >= top; row-- {
                result = append(result, matrix[row][left])
            }
            left++
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Direction Array Approach**
```go
func spiralOrderDirection(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }
    
    m, n := len(matrix), len(matrix[0])
    result := make([]int, 0, m*n)
    
    directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
    direction := 0
    row, col := 0, 0
    visited := make([][]bool, m)
    for i := range visited {
        visited[i] = make([]bool, n)
    }
    
    for i := 0; i < m*n; i++ {
        result = append(result, matrix[row][col])
        visited[row][col] = true
        
        nextRow := row + directions[direction][0]
        nextCol := col + directions[direction][1]
        
        if nextRow < 0 || nextRow >= m || nextCol < 0 || nextCol >= n || visited[nextRow][nextCol] {
            direction = (direction + 1) % 4
            nextRow = row + directions[direction][0]
            nextCol = col + directions[direction][1]
        }
        
        row, col = nextRow, nextCol
    }
    
    return result
}
```

### Notes / Variations

#### **Related Problems**
- **Spiral Matrix II**: Generate spiral matrix
- **Spiral Matrix III**: Spiral matrix with different starting point
- **Rotate Image**: Rotate matrix 90 degrees
- **Set Matrix Zeroes**: Set entire row/column to zero

#### **ICPC Insights**
- **Boundary Management**: Carefully manage boundaries
- **Direction Handling**: Handle direction changes properly
- **Edge Cases**: Handle single row/column matrices
- **Space Efficiency**: Use O(1) space approach
