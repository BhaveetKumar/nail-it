# Word Break

### Problem
Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

**Example:**
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
```

### Golang Solution

```go
func wordBreak(s string, wordDict []string) bool {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    dp := make([]bool, len(s)+1)
    dp[0] = true
    
    for i := 1; i <= len(s); i++ {
        for j := 0; j < i; j++ {
            if dp[j] && wordSet[s[j:i]] {
                dp[i] = true
                break
            }
        }
    }
    
    return dp[len(s)]
}
```

### Alternative Solutions

#### **DFS with Memoization**
```go
func wordBreakDFS(s string, wordDict []string) bool {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    memo := make(map[int]bool)
    
    var dfs func(int) bool
    dfs = func(start int) bool {
        if start == len(s) {
            return true
        }
        
        if val, exists := memo[start]; exists {
            return val
        }
        
        for end := start + 1; end <= len(s); end++ {
            if wordSet[s[start:end]] && dfs(end) {
                memo[start] = true
                return true
            }
        }
        
        memo[start] = false
        return false
    }
    
    return dfs(0)
}
```

#### **BFS Approach**
```go
func wordBreakBFS(s string, wordDict []string) bool {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    queue := []int{0}
    visited := make([]bool, len(s))
    
    for len(queue) > 0 {
        start := queue[0]
        queue = queue[1:]
        
        if visited[start] {
            continue
        }
        
        visited[start] = true
        
        for end := start + 1; end <= len(s); end++ {
            if wordSet[s[start:end]] {
                if end == len(s) {
                    return true
                }
                queue = append(queue, end)
            }
        }
    }
    
    return false
}
```

### Complexity
- **Time Complexity:** O(n²) for DP, O(n²) for DFS with memo, O(n²) for BFS
- **Space Complexity:** O(n)
