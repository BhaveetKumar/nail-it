# Unique Paths

### Problem
There is a robot on an `m x n` grid. The robot is initially located at the top-left corner (i.e., `grid[0][0]`). The robot tries to move to the bottom-right corner (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

Given the two integers `m` and `n`, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

**Example:**
```
Input: m = 3, n = 7
Output: 28

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
```

### Golang Solution

```go
func uniquePaths(m int, n int) int {
    dp := make([][]int, m)
    for i := range dp {
        dp[i] = make([]int, n)
    }
    
    // Initialize first row and column
    for i := 0; i < m; i++ {
        dp[i][0] = 1
    }
    for j := 0; j < n; j++ {
        dp[0][j] = 1
    }
    
    // Fill the DP table
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
        }
    }
    
    return dp[m-1][n-1]
}
```

### Alternative Solutions

#### **Space Optimized**
```go
func uniquePathsOptimized(m int, n int) int {
    prev := make([]int, n)
    for i := range prev {
        prev[i] = 1
    }
    
    for i := 1; i < m; i++ {
        curr := make([]int, n)
        curr[0] = 1
        
        for j := 1; j < n; j++ {
            curr[j] = curr[j-1] + prev[j]
        }
        
        prev = curr
    }
    
    return prev[n-1]
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(m × n) for DP, O(n) for optimized
