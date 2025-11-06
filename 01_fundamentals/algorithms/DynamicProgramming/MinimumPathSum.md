---
# Auto-generated front matter
Title: Minimumpathsum
LastUpdated: 2025-11-06T20:45:58.745484
Tags: []
Status: draft
---

# Minimum Path Sum

### Problem
Given a `m x n` grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

**Example:**
```
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Input: grid = [[1,2,3],[4,5,6]]
Output: 12
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

#### **Using Extra Space**
```go
func minPathSumExtraSpace(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    dp := make([][]int, m)
    
    for i := range dp {
        dp[i] = make([]int, n)
    }
    
    dp[0][0] = grid[0][0]
    
    // Initialize first row
    for j := 1; j < n; j++ {
        dp[0][j] = dp[0][j-1] + grid[0][j]
    }
    
    // Initialize first column
    for i := 1; i < m; i++ {
        dp[i][0] = dp[i-1][0] + grid[i][0]
    }
    
    // Fill the DP table
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        }
    }
    
    return dp[m-1][n-1]
}
```

#### **Space Optimized**
```go
func minPathSumOptimized(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    prev := make([]int, n)
    curr := make([]int, n)
    
    prev[0] = grid[0][0]
    
    // Initialize first row
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

#### **Recursive with Memoization**
```go
func minPathSumMemo(grid [][]int) int {
    m, n := len(grid), len(grid[0])
    memo := make([][]int, m)
    
    for i := range memo {
        memo[i] = make([]int, n)
        for j := range memo[i] {
            memo[i][j] = -1
        }
    }
    
    return minPathSumHelper(grid, 0, 0, memo)
}

func minPathSumHelper(grid [][]int, i, j int, memo [][]int) int {
    m, n := len(grid), len(grid[0])
    
    if i == m-1 && j == n-1 {
        return grid[i][j]
    }
    
    if memo[i][j] != -1 {
        return memo[i][j]
    }
    
    result := math.MaxInt32
    
    if i+1 < m {
        result = min(result, minPathSumHelper(grid, i+1, j, memo))
    }
    
    if j+1 < n {
        result = min(result, minPathSumHelper(grid, i, j+1, memo))
    }
    
    memo[i][j] = result + grid[i][j]
    return memo[i][j]
}
```

#### **Return Path**
```go
func minPathSumWithPath(grid [][]int) (int, [][]int) {
    m, n := len(grid), len(grid[0])
    dp := make([][]int, m)
    path := make([][]int, m)
    
    for i := range dp {
        dp[i] = make([]int, n)
        path[i] = make([]int, n)
    }
    
    dp[0][0] = grid[0][0]
    path[0][0] = 0 // 0: from top, 1: from left
    
    // Initialize first row
    for j := 1; j < n; j++ {
        dp[0][j] = dp[0][j-1] + grid[0][j]
        path[0][j] = 1
    }
    
    // Initialize first column
    for i := 1; i < m; i++ {
        dp[i][0] = dp[i-1][0] + grid[i][0]
        path[i][0] = 0
    }
    
    // Fill the DP table
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if dp[i-1][j] < dp[i][j-1] {
                dp[i][j] = dp[i-1][j] + grid[i][j]
                path[i][j] = 0
            } else {
                dp[i][j] = dp[i][j-1] + grid[i][j]
                path[i][j] = 1
            }
        }
    }
    
    // Reconstruct path
    var resultPath [][]int
    i, j := m-1, n-1
    
    for i > 0 || j > 0 {
        resultPath = append([][]int{{i, j}}, resultPath...)
        
        if path[i][j] == 0 {
            i--
        } else {
            j--
        }
    }
    
    resultPath = append([][]int{{0, 0}}, resultPath...)
    
    return dp[m-1][n-1], resultPath
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1) for in-place, O(m × n) for extra space