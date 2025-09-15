# Climbing Stairs

### Problem
You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

**Example:**
```
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

### Golang Solution

```go
func climbStairs(n int) int {
    if n <= 2 {
        return n
    }
    
    prev2 := 1
    prev1 := 2
    
    for i := 3; i <= n; i++ {
        current := prev1 + prev2
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}
```

### Alternative Solutions

#### **Recursive with Memoization**
```go
func climbStairsMemo(n int) int {
    memo := make(map[int]int)
    return climbStairsHelper(n, memo)
}

func climbStairsHelper(n int, memo map[int]int) int {
    if n <= 2 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    memo[n] = climbStairsHelper(n-1, memo) + climbStairsHelper(n-2, memo)
    return memo[n]
}
```

#### **Using Array DP**
```go
func climbStairsDP(n int) int {
    if n <= 2 {
        return n
    }
    
    dp := make([]int, n+1)
    dp[1] = 1
    dp[2] = 2
    
    for i := 3; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    
    return dp[n]
}
```

#### **Return All Paths**
```go
func climbStairsPaths(n int) [][]int {
    var result [][]int
    
    var dfs func(int, []int)
    dfs = func(remaining int, path []int) {
        if remaining == 0 {
            resultPath := make([]int, len(path))
            copy(resultPath, path)
            result = append(result, resultPath)
            return
        }
        
        if remaining >= 1 {
            dfs(remaining-1, append(path, 1))
        }
        
        if remaining >= 2 {
            dfs(remaining-2, append(path, 2))
        }
    }
    
    dfs(n, []int{})
    return result
}
```

#### **Generalized for Any Step Sizes**
```go
func climbStairsGeneralized(n int, steps []int) int {
    if n == 0 {
        return 1
    }
    
    dp := make([]int, n+1)
    dp[0] = 1
    
    for i := 1; i <= n; i++ {
        for _, step := range steps {
            if i >= step {
                dp[i] += dp[i-step]
            }
        }
    }
    
    return dp[n]
}
```

#### **Return with Statistics**
```go
type ClimbingStats struct {
    TotalWays    int
    PathsWith1   int
    PathsWith2   int
    MinSteps     int
    MaxSteps     int
}

func climbStairsStats(n int) ClimbingStats {
    paths := climbStairsPaths(n)
    
    pathsWith1 := 0
    pathsWith2 := 0
    minSteps := math.MaxInt32
    maxSteps := 0
    
    for _, path := range paths {
        has1 := false
        has2 := false
        
        for _, step := range path {
            if step == 1 {
                has1 = true
            } else if step == 2 {
                has2 = true
            }
        }
        
        if has1 {
            pathsWith1++
        }
        if has2 {
            pathsWith2++
        }
        
        if len(path) < minSteps {
            minSteps = len(path)
        }
        if len(path) > maxSteps {
            maxSteps = len(path)
        }
    }
    
    return ClimbingStats{
        TotalWays:  len(paths),
        PathsWith1: pathsWith1,
        PathsWith2: pathsWith2,
        MinSteps:   minSteps,
        MaxSteps:   maxSteps,
    }
}
```

#### **Using Matrix Exponentiation**
```go
func climbStairsMatrix(n int) int {
    if n <= 2 {
        return n
    }
    
    // Base matrix: [[1, 1], [1, 0]]
    base := [][]int{{1, 1}, {1, 0}}
    
    // Calculate base^(n-1)
    result := matrixPower(base, n-1)
    
    // Result is result[0][0] * 2 + result[0][1] * 1
    return result[0][0]*2 + result[0][1]*1
}

func matrixPower(matrix [][]int, power int) [][]int {
    if power == 1 {
        return matrix
    }
    
    if power%2 == 0 {
        half := matrixPower(matrix, power/2)
        return matrixMultiply(half, half)
    }
    
    return matrixMultiply(matrix, matrixPower(matrix, power-1))
}

func matrixMultiply(a, b [][]int) [][]int {
    result := make([][]int, 2)
    for i := range result {
        result[i] = make([]int, 2)
    }
    
    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            for k := 0; k < 2; k++ {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for iterative, O(2^n) for naive recursive
- **Space Complexity:** O(1) for iterative, O(n) for DP array