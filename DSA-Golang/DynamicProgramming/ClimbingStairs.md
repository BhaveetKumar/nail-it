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

**Constraints:**
- 1 ≤ n ≤ 45

### Explanation

#### **Dynamic Programming Approach**
- This is essentially the Fibonacci sequence
- dp[i] = dp[i-1] + dp[i-2]
- Base cases: dp[1] = 1, dp[2] = 2
- Time Complexity: O(n)
- Space Complexity: O(1)

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

### Notes / Variations

#### **Related Problems**
- **Fibonacci Number**: Calculate nth Fibonacci number
- **House Robber**: Maximum money without adjacent houses
- **Min Cost Climbing Stairs**: Minimum cost to reach top
- **Unique Paths**: Number of paths in grid
