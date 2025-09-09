# Dynamic Programming Problems

Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems. It's particularly useful for optimization problems where we need to find the best solution among many possibilities.

## Key Concepts

### DP Characteristics
1. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems are solved multiple times
3. **Memoization**: Store results of subproblems to avoid recomputation
4. **Tabulation**: Build solution bottom-up using a table

### DP Patterns
1. **1D DP**: Problems with single dimension (Fibonacci, Climbing Stairs)
2. **2D DP**: Problems with two dimensions (Grid paths, Edit Distance)
3. **Knapsack**: Optimization problems with constraints
4. **Longest Common Subsequence**: String comparison problems
5. **Matrix Chain Multiplication**: Optimization problems
6. **Coin Change**: Combinatorial problems

## Problems

### 1. [Fibonacci Sequence](./Fibonacci.md)
Calculate the nth Fibonacci number using DP.

### 2. [Climbing Stairs](./ClimbingStairs.md)
Find number of ways to climb n stairs with 1 or 2 steps.

### 3. [House Robber](./HouseRobber.md)
Maximize money robbed from houses without robbing adjacent ones.

### 4. [Longest Common Subsequence](./LongestCommonSubsequence.md)
Find the longest common subsequence between two strings.

### 5. [Edit Distance](./EditDistance.md)
Find minimum operations to convert one string to another.

### 6. [Coin Change](./CoinChange.md)
Find minimum number of coins to make a target amount.

### 7. [Knapsack Problem](./Knapsack.md)
Maximize value in knapsack with weight constraint.

### 8. [Longest Increasing Subsequence](./LongestIncreasingSubsequence.md)
Find length of longest increasing subsequence.

### 9. [Word Break](./WordBreak.md)
Check if string can be segmented into dictionary words.

### 10. [Unique Paths](./UniquePaths.md)
Find number of unique paths in a grid.

## Time & Space Complexity

| Problem | Time Complexity | Space Complexity |
|---------|----------------|------------------|
| Fibonacci | O(n) | O(1) |
| Climbing Stairs | O(n) | O(1) |
| House Robber | O(n) | O(1) |
| LCS | O(m×n) | O(m×n) |
| Edit Distance | O(m×n) | O(m×n) |
| Coin Change | O(amount×coins) | O(amount) |
| Knapsack | O(n×W) | O(n×W) |
| LIS | O(n²) | O(n) |
| Word Break | O(n²) | O(n) |
| Unique Paths | O(m×n) | O(m×n) |

Where:
- n = input size
- m, n = string/array lengths
- W = knapsack capacity

## DP Techniques

### 1. Memoization (Top-Down)
```javascript
function memoizedDP(n, memo = {}) {
    if (n in memo) return memo[n];
    if (n <= 1) return n;
    
    memo[n] = memoizedDP(n - 1, memo) + memoizedDP(n - 2, memo);
    return memo[n];
}
```

### 2. Tabulation (Bottom-Up)
```javascript
function tabulatedDP(n) {
    const dp = Array(n + 1).fill(0);
    dp[0] = 0;
    dp[1] = 1;
    
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}
```

### 3. Space Optimization
```javascript
function optimizedDP(n) {
    if (n <= 1) return n;
    
    let prev2 = 0, prev1 = 1;
    for (let i = 2; i <= n; i++) {
        const current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

## Tips for DP Problems

1. **Identify the Pattern**: Recognize if it's a DP problem
2. **Define State**: What does dp[i] represent?
3. **Find Recurrence**: How to compute dp[i] from previous states?
4. **Base Cases**: What are the initial values?
5. **Optimize Space**: Can we reduce space complexity?
6. **Trace Back**: How to reconstruct the solution?

## Common Mistakes

1. **Missing Base Cases**: Not handling edge cases properly
2. **Wrong State Definition**: Incorrect dp[i] meaning
3. **Incorrect Recurrence**: Wrong formula for computing states
4. **Index Errors**: Off-by-one errors in array access
5. **Space Inefficiency**: Not optimizing space when possible
