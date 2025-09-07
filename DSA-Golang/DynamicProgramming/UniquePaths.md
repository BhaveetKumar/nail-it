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
    
    // Initialize first row and first column
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
    curr := make([]int, n)
    
    // Initialize first row
    for j := 0; j < n; j++ {
        prev[j] = 1
    }
    
    for i := 1; i < m; i++ {
        curr[0] = 1
        for j := 1; j < n; j++ {
            curr[j] = curr[j-1] + prev[j]
        }
        prev, curr = curr, prev
    }
    
    return prev[n-1]
}
```

#### **Mathematical Approach**
```go
func uniquePathsMath(m int, n int) int {
    // Total moves = (m-1) + (n-1) = m+n-2
    // We need to choose (m-1) down moves or (n-1) right moves
    // Answer = C(m+n-2, m-1) = C(m+n-2, n-1)
    
    total := m + n - 2
    k := m - 1
    if k > total-k {
        k = total - k
    }
    
    result := 1
    for i := 0; i < k; i++ {
        result = result * (total - i) / (i + 1)
    }
    
    return result
}
```

#### **Recursive with Memoization**
```go
func uniquePathsMemo(m int, n int) int {
    memo := make([][]int, m)
    for i := range memo {
        memo[i] = make([]int, n)
        for j := range memo[i] {
            memo[i][j] = -1
        }
    }
    
    return uniquePathsHelper(0, 0, m, n, memo)
}

func uniquePathsHelper(i, j, m, n int, memo [][]int) int {
    if i == m-1 || j == n-1 {
        return 1
    }
    
    if memo[i][j] != -1 {
        return memo[i][j]
    }
    
    memo[i][j] = uniquePathsHelper(i+1, j, m, n, memo) + uniquePathsHelper(i, j+1, m, n, memo)
    return memo[i][j]
}
```

### Complexity
- **Time Complexity:** O(m × n) for DP, O(m + n) for math
- **Space Complexity:** O(m × n) for DP, O(n) for optimized, O(1) for math
