---
# Auto-generated front matter
Title: Uniquepathsii
LastUpdated: 2025-11-06T20:45:58.745282
Tags: []
Status: draft
---

# Unique Paths II

### Problem
You are given an `m x n` integer array `grid`. There is a robot initially located at the top-left corner (i.e., `grid[0][0]`). The robot tries to move to the bottom-right corner (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

An obstacle and space are marked as `1` or `0` respectively in `grid`. A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

**Example:**
```
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
```

### Golang Solution

```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
    m, n := len(obstacleGrid), len(obstacleGrid[0])
    
    if obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1 {
        return 0
    }
    
    dp := make([][]int, m)
    for i := range dp {
        dp[i] = make([]int, n)
    }
    
    // Initialize first row
    for j := 0; j < n && obstacleGrid[0][j] == 0; j++ {
        dp[0][j] = 1
    }
    
    // Initialize first column
    for i := 0; i < m && obstacleGrid[i][0] == 0; i++ {
        dp[i][0] = 1
    }
    
    // Fill the DP table
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            if obstacleGrid[i][j] == 0 {
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            }
        }
    }
    
    return dp[m-1][n-1]
}
```

### Alternative Solutions

#### **Space Optimized**
```go
func uniquePathsWithObstaclesOptimized(obstacleGrid [][]int) int {
    m, n := len(obstacleGrid), len(obstacleGrid[0])
    
    if obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1 {
        return 0
    }
    
    prev := make([]int, n)
    curr := make([]int, n)
    
    // Initialize first row
    for j := 0; j < n && obstacleGrid[0][j] == 0; j++ {
        prev[j] = 1
    }
    
    for i := 1; i < m; i++ {
        // Initialize first column
        if obstacleGrid[i][0] == 0 {
            curr[0] = prev[0]
        } else {
            curr[0] = 0
        }
        
        for j := 1; j < n; j++ {
            if obstacleGrid[i][j] == 0 {
                curr[j] = curr[j-1] + prev[j]
            } else {
                curr[j] = 0
            }
        }
        
        prev, curr = curr, prev
    }
    
    return prev[n-1]
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(m × n) for DP, O(n) for optimized
