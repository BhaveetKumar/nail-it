# Dynamic Programming Pattern

> **Master dynamic programming techniques with Go implementations**

## 📋 Problems

### **1D DP**
- [Climbing Stairs](./ClimbingStairs.md) - Fibonacci sequence variation
- [House Robber](./HouseRobber.md) - Maximum sum with constraints
- [Longest Increasing Subsequence](./LongestIncreasingSubsequence.md) - LIS with O(n log n) solution
- [Word Break](./WordBreak.md) - String segmentation problem
- [Decode Ways](./DecodeWays.md) - String to number conversion

### **2D DP**
- [Unique Paths](./UniquePaths.md) - Grid traversal counting
- [Minimum Path Sum](./MinimumPathSum.md) - Grid optimization
- [Longest Common Subsequence](./LongestCommonSubsequence.md) - String comparison
- [Edit Distance](./EditDistance.md) - String transformation
- [Knapsack](./Knapsack.md) - Classic optimization problem

### **Advanced DP**
- [Coin Change](./CoinChange.md) - Minimum coins problem
- [Palindrome Partitioning](./PalindromePartitioning.md) - String partitioning
- [Regular Expression Matching](./RegularExpressionMatching.md) - Pattern matching
- [Wildcard Matching](./WildcardMatching.md) - Advanced pattern matching
- [Maximum Product Subarray](./MaximumProductSubarray.md) - Array optimization

---

## 🎯 Key Concepts

### **Dynamic Programming Principles**
1. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times
3. **Memoization**: Store results of subproblems to avoid recomputation
4. **Tabulation**: Build solution bottom-up using table

### **DP Patterns**
- **1D DP**: Linear problems (Fibonacci, House Robber)
- **2D DP**: Grid/matrix problems (Unique Paths, LCS)
- **Interval DP**: Problems on ranges (Matrix Chain Multiplication)
- **Tree DP**: Problems on trees (Binary Tree Cameras)
- **State Machine DP**: Problems with states (Stock Trading)

### **Implementation Approaches**
- **Top-Down (Memoization)**: Recursive with memoization
- **Bottom-Up (Tabulation)**: Iterative with table
- **Space Optimization**: Reduce space complexity

---

## 🛠️ Go-Specific Tips

### **Memoization with Maps**
```go
// Global memoization map
var memo = make(map[int]int)

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    memo[n] = fibonacci(n-1) + fibonacci(n-2)
    return memo[n]
}
```

### **2D DP Table**
```go
// Create 2D slice for DP table
dp := make([][]int, m)
for i := range dp {
    dp[i] = make([]int, n)
}

// Initialize base cases
dp[0][0] = 1

// Fill the table
for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        // DP transition
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
    }
}
```

### **Space Optimization**
```go
// Instead of 2D table, use 1D array
prev := make([]int, n)
curr := make([]int, n)

for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        curr[j] = prev[j] + curr[j-1]
    }
    prev, curr = curr, prev
}
```

### **Common DP Patterns in Go**
```go
// 1. Fibonacci pattern
func dp(n int) int {
    if n <= 1 {
        return n
    }
    
    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    return b
}

// 2. LIS pattern
func lis(nums []int) int {
    dp := make([]int, len(nums))
    for i := range dp {
        dp[i] = 1
    }
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
    }
    
    return max(dp...)
}
```

---

## 🎯 Interview Tips

### **How to Identify DP Problems**
1. **Optimization Problem**: Find minimum/maximum/count
2. **Decision Problem**: Yes/No with constraints
3. **Overlapping Subproblems**: Same subproblems appear multiple times
4. **Optimal Substructure**: Optimal solution contains optimal subproblems

### **DP Problem Solving Steps**
1. **Identify the problem type** (1D, 2D, interval, etc.)
2. **Define the state** (what does dp[i] represent?)
3. **Find the recurrence relation** (how to compute dp[i] from previous states?)
4. **Identify base cases** (initial values)
5. **Choose implementation approach** (top-down or bottom-up)
6. **Optimize space** (if possible)

### **Common Mistakes to Avoid**
- **Incorrect state definition**: Make sure dp[i] represents what you think it does
- **Wrong base cases**: Initialize correctly
- **Off-by-one errors**: Be careful with array indices
- **Memory issues**: Use space optimization for large inputs
