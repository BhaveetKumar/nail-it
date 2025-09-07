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
    
    prev2 := 1 // dp[i-2]
    prev1 := 2 // dp[i-1]
    
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

#### **Using Array**
```go
func climbStairsArray(n int) int {
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

#### **Mathematical Formula**
```go
func climbStairsMath(n int) int {
    if n <= 2 {
        return n
    }
    
    // Fibonacci sequence: F(n) = F(n-1) + F(n-2)
    // Using Binet's formula approximation
    sqrt5 := math.Sqrt(5)
    phi := (1 + sqrt5) / 2
    psi := (1 - sqrt5) / 2
    
    result := (math.Pow(phi, float64(n+1)) - math.Pow(psi, float64(n+1))) / sqrt5
    return int(math.Round(result))
}
```

#### **Matrix Exponentiation**
```go
func climbStairsMatrix(n int) int {
    if n <= 2 {
        return n
    }
    
    // [F(n+1)]   [1 1]^n   [F(1)]
    // [F(n)  ] = [1 0]   * [F(0)]
    
    matrix := [][]int{{1, 1}, {1, 0}}
    result := matrixPower(matrix, n)
    
    return result[0][0]
}

func matrixPower(matrix [][]int, n int) [][]int {
    if n == 1 {
        return matrix
    }
    
    if n%2 == 0 {
        half := matrixPower(matrix, n/2)
        return matrixMultiply(half, half)
    }
    
    return matrixMultiply(matrix, matrixPower(matrix, n-1))
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
- **Time Complexity:** O(n) for DP, O(log n) for matrix exponentiation
- **Space Complexity:** O(1) for optimized DP, O(n) for array