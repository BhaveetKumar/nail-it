---
# Auto-generated front matter
Title: Fibonaccisequence
LastUpdated: 2025-11-06T20:45:58.747828
Tags: []
Status: draft
---

# Fibonacci Sequence

### Problem
The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1.

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.

Given n, calculate F(n).

**Example:**
```
Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.

Input: n = 3
Output: 2
Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.
```

### Golang Solution

```go
func fib(n int) int {
    if n <= 1 {
        return n
    }
    
    prev2 := 0
    prev1 := 1
    
    for i := 2; i <= n; i++ {
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
func fibMemo(n int) int {
    memo := make(map[int]int)
    return fibHelper(n, memo)
}

func fibHelper(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    memo[n] = fibHelper(n-1, memo) + fibHelper(n-2, memo)
    return memo[n]
}
```

### Complexity
- **Time Complexity:** O(n) for iterative, O(n) for memoized recursive
- **Space Complexity:** O(1) for iterative, O(n) for memoized recursive
