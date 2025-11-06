---
# Auto-generated front matter
Title: Decodeways
LastUpdated: 2025-11-06T20:45:58.684874
Tags: []
Status: draft
---

# Decode Ways

### Problem
A message containing letters from `A-Z` can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, `"11106"` can be mapped into:

- "AAJF" with the grouping (1 1 10 6)
- "KJF" with the grouping (11 10 6)

Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string `s` containing only digits, return the number of ways to decode it.

**Example:**
```
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).

Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
```

### Golang Solution

```go
func numDecodings(s string) int {
    if len(s) == 0 || s[0] == '0' {
        return 0
    }
    
    n := len(s)
    dp := make([]int, n+1)
    dp[0] = 1
    dp[1] = 1
    
    for i := 2; i <= n; i++ {
        // Single digit
        if s[i-1] != '0' {
            dp[i] += dp[i-1]
        }
        
        // Two digits
        twoDigit := int(s[i-2]-'0')*10 + int(s[i-1]-'0')
        if twoDigit >= 10 && twoDigit <= 26 {
            dp[i] += dp[i-2]
        }
    }
    
    return dp[n]
}
```

### Alternative Solutions

#### **Space Optimized**
```go
func numDecodingsOptimized(s string) int {
    if len(s) == 0 || s[0] == '0' {
        return 0
    }
    
    prev2 := 1 // dp[i-2]
    prev1 := 1 // dp[i-1]
    
    for i := 1; i < len(s); i++ {
        current := 0
        
        // Single digit
        if s[i] != '0' {
            current += prev1
        }
        
        // Two digits
        twoDigit := int(s[i-1]-'0')*10 + int(s[i]-'0')
        if twoDigit >= 10 && twoDigit <= 26 {
            current += prev2
        }
        
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n) for DP, O(1) for optimized
