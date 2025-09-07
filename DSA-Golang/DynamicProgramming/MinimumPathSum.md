# Minimum Path Sum

### Problem
Given a `m x n` `grid` filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

**Example:**
```
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
```

### Golang Solution

```go
func minPathSum(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    
    // Initialize first row
    for j := 1; j < n; j++ {
        grid[0][j] += grid[0][j-1]
    }
    
    // Initialize first column
    for i := 1; i < m; i++ {
        grid[i][0] += grid[i-1][0]
    }
    
    // Fill the DP table
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        }
    }
    
    return grid[m-1][n-1]
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Space Optimized**
```go
func minPathSumOptimized(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    prev := make([]int, n)
    curr := make([]int, n)
    
    prev[0] = grid[0][0]
    for j := 1; j < n; j++ {
        prev[j] = prev[j-1] + grid[0][j]
    }
    
    for i := 1; i < m; i++ {
        curr[0] = prev[0] + grid[i][0]
        for j := 1; j < n; j++ {
            curr[j] = min(prev[j], curr[j-1]) + grid[i][j]
        }
        prev, curr = curr, prev
    }
    
    return prev[n-1]
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1) for in-place, O(n) for optimized
