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

#### **Recursive with Memoization**
```go
func wordBreakMemo(s string, wordDict []string) bool {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    memo := make(map[int]bool)
    return wordBreakHelper(s, 0, wordSet, memo)
}

func wordBreakHelper(s string, start int, wordSet map[string]bool, memo map[int]bool) bool {
    if start == len(s) {
        return true
    }
    
    if val, exists := memo[start]; exists {
        return val
    }
    
    for end := start + 1; end <= len(s); end++ {
        if wordSet[s[start:end]] && wordBreakHelper(s, end, wordSet, memo) {
            memo[start] = true
            return true
        }
    }
    
    memo[start] = false
    return false
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
    visited := make(map[int]bool)
    
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

#### **Return All Possible Sentences**
```go
func wordBreakAll(s string, wordDict []string) []string {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    memo := make(map[int][]string)
    return wordBreakAllHelper(s, 0, wordSet, memo)
}

func wordBreakAllHelper(s string, start int, wordSet map[string]bool, memo map[int][]string) []string {
    if start == len(s) {
        return []string{""}
    }
    
    if val, exists := memo[start]; exists {
        return val
    }
    
    var result []string
    
    for end := start + 1; end <= len(s); end++ {
        word := s[start:end]
        if wordSet[word] {
            subResults := wordBreakAllHelper(s, end, wordSet, memo)
            
            for _, subResult := range subResults {
                if subResult == "" {
                    result = append(result, word)
                } else {
                    result = append(result, word+" "+subResult)
                }
            }
        }
    }
    
    memo[start] = result
    return result
}
```

#### **Return Minimum Number of Words**
```go
func wordBreakMinWords(s string, wordDict []string) int {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    dp := make([]int, len(s)+1)
    for i := 1; i <= len(s); i++ {
        dp[i] = math.MaxInt32
    }
    
    for i := 1; i <= len(s); i++ {
        for j := 0; j < i; j++ {
            if dp[j] != math.MaxInt32 && wordSet[s[j:i]] {
                dp[i] = min(dp[i], dp[j]+1)
            }
        }
    }
    
    if dp[len(s)] == math.MaxInt32 {
        return -1
    }
    
    return dp[len(s)]
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n²) for DP, O(n²) for BFS
- **Space Complexity:** O(n) for DP, O(n) for BFS