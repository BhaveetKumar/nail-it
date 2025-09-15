# Edit Distance

### Problem
Given two strings `word1` and `word2`, return the minimum number of operations required to convert `word1` to `word2`.

You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

**Example:**
```
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Input: word1 = "intention", word2 = "execution"
Output: 5
```

### Golang Solution

```go
func minDistance(word1 string, word2 string) int {
    m, n := len(word1), len(word2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Initialize base cases
    for i := 0; i <= m; i++ {
        dp[i][0] = i
    }
    for j := 0; j <= n; j++ {
        dp[0][j] = j
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
    }
    
    return dp[m][n]
}

func min(a, b, c int) int {
    if a < b && a < c {
        return a
    } else if b < c {
        return b
    }
    return c
}
```

### Alternative Solutions

#### **Space Optimized**
```go
func minDistanceOptimized(word1 string, word2 string) int {
    if len(word1) < len(word2) {
        word1, word2 = word2, word1
    }
    
    m, n := len(word1), len(word2)
    prev := make([]int, n+1)
    curr := make([]int, n+1)
    
    for j := 0; j <= n; j++ {
        prev[j] = j
    }
    
    for i := 1; i <= m; i++ {
        curr[0] = i
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                curr[j] = prev[j-1]
            } else {
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            }
        }
        prev, curr = curr, prev
    }
    
    return prev[n]
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(m × n) for DP, O(min(m, n)) for optimized
