---
# Auto-generated front matter
Title: Uniquepaths
LastUpdated: 2025-11-06T20:45:58.747666
Tags: []
Status: draft
---

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
    if m > n {
        m, n = n, m
    }
    
    dp := make([]int, m)
    for i := range dp {
        dp[i] = 1
    }
    
    for j := 1; j < n; j++ {
        for i := 1; i < m; i++ {
            dp[i] += dp[i-1]
        }
    }
    
    return dp[m-1]
}
```

#### **Mathematical Formula**
```go
func uniquePathsMath(m int, n int) int {
    // Total moves = (m-1) + (n-1) = m+n-2
    // We need to choose (m-1) down moves or (n-1) right moves
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
    
    return uniquePathsHelper(m-1, n-1, memo)
}

func uniquePathsHelper(i, j int, memo [][]int) int {
    if i == 0 || j == 0 {
        return 1
    }
    
    if memo[i][j] != -1 {
        return memo[i][j]
    }
    
    memo[i][j] = uniquePathsHelper(i-1, j, memo) + uniquePathsHelper(i, j-1, memo)
    return memo[i][j]
}
```

#### **Return All Paths**
```go
func uniquePathsAll(m int, n int) [][]string {
    var paths [][]string
    
    var dfs func(int, int, []string)
    dfs = func(i, j int, path []string) {
        if i == m-1 && j == n-1 {
            resultPath := make([]string, len(path))
            copy(resultPath, path)
            paths = append(paths, resultPath)
            return
        }
        
        if i < m-1 {
            dfs(i+1, j, append(path, "Down"))
        }
        
        if j < n-1 {
            dfs(i, j+1, append(path, "Right"))
        }
    }
    
    dfs(0, 0, []string{})
    return paths
}
```

#### **Return Path Statistics**
```go
type PathStats struct {
    TotalPaths    int
    MinLength     int
    MaxLength     int
    AvgLength     float64
    DownMoves     int
    RightMoves    int
}

func uniquePathsStats(m int, n int) PathStats {
    paths := uniquePathsAll(m, n)
    
    if len(paths) == 0 {
        return PathStats{}
    }
    
    minLength := len(paths[0])
    maxLength := len(paths[0])
    totalLength := 0
    downMoves := 0
    rightMoves := 0
    
    for _, path := range paths {
        length := len(path)
        totalLength += length
        
        if length < minLength {
            minLength = length
        }
        if length > maxLength {
            maxLength = length
        }
        
        for _, move := range path {
            if move == "Down" {
                downMoves++
            } else {
                rightMoves++
            }
        }
    }
    
    return PathStats{
        TotalPaths: len(paths),
        MinLength:  minLength,
        MaxLength:  maxLength,
        AvgLength:  float64(totalLength) / float64(len(paths)),
        DownMoves:  downMoves,
        RightMoves: rightMoves,
    }
}
```

#### **With Obstacles**
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
    for j := 0; j < n; j++ {
        if obstacleGrid[0][j] == 1 {
            break
        }
        dp[0][j] = 1
    }
    
    // Initialize first column
    for i := 0; i < m; i++ {
        if obstacleGrid[i][0] == 1 {
            break
        }
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

### Complexity
- **Time Complexity:** O(m×n) for DP, O(m+n) for mathematical
- **Space Complexity:** O(m×n) for DP, O(min(m,n)) for optimized