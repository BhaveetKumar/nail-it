---
# Auto-generated front matter
Title: Wordbreak
LastUpdated: 2025-11-06T20:45:58.689198
Tags: []
Status: draft
---

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
    return dfs(s, 0, wordSet, memo)
}

func dfs(s string, start int, wordSet map[string]bool, memo map[int]bool) bool {
    if start == len(s) {
        return true
    }
    
    if result, exists := memo[start]; exists {
        return result
    }
    
    for end := start + 1; end <= len(s); end++ {
        if wordSet[s[start:end]] && dfs(s, end, wordSet, memo) {
            memo[start] = true
            return true
        }
    }
    
    memo[start] = false
    return false
}
```

### Complexity
- **Time Complexity:** O(n²) for DP, O(n²) for DFS with memoization
- **Space Complexity:** O(n) for DP, O(n) for DFS with memoization
