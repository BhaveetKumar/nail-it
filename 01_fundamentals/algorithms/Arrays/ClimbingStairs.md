---
# Auto-generated front matter
Title: Climbingstairs
LastUpdated: 2025-11-06T20:45:58.726449
Tags: []
Status: draft
---

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

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for memoized
