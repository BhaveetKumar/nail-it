---
# Auto-generated front matter
Title: Longestsubstringwithoutrepeatingcharacters
LastUpdated: 2025-11-06T20:45:58.711903
Tags: []
Status: draft
---

# Longest Substring Without Repeating Characters

### Problem
Given a string `s`, find the length of the longest substring without repeating characters.

**Example:**
```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

### Golang Solution

```go
func lengthOfLongestSubstring(s string) int {
    charIndex := make(map[byte]int)
    maxLen := 0
    left := 0
    
    for right := 0; right < len(s); right++ {
        if index, exists := charIndex[s[right]]; exists && index >= left {
            left = index + 1
        }
        
        charIndex[s[right]] = right
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Using Array for ASCII Characters**
```go
func lengthOfLongestSubstringArray(s string) int {
    charIndex := make([]int, 128) // ASCII characters
    for i := range charIndex {
        charIndex[i] = -1
    }
    
    maxLen := 0
    left := 0
    
    for right := 0; right < len(s); right++ {
        if charIndex[s[right]] >= left {
            left = charIndex[s[right]] + 1
        }
        
        charIndex[s[right]] = right
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

#### **Using Set**
```go
func lengthOfLongestSubstringSet(s string) int {
    charSet := make(map[byte]bool)
    maxLen := 0
    left := 0
    
    for right := 0; right < len(s); right++ {
        for charSet[s[right]] {
            delete(charSet, s[left])
            left++
        }
        
        charSet[s[right]] = true
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

#### **Brute Force**
```go
func lengthOfLongestSubstringBruteForce(s string) int {
    maxLen := 0
    
    for i := 0; i < len(s); i++ {
        charSet := make(map[byte]bool)
        currentLen := 0
        
        for j := i; j < len(s); j++ {
            if charSet[s[j]] {
                break
            }
            charSet[s[j]] = true
            currentLen++
        }
        
        maxLen = max(maxLen, currentLen)
    }
    
    return maxLen
}
```

#### **Return the Actual Substring**
```go
func longestSubstringWithoutRepeating(s string) string {
    charIndex := make(map[byte]int)
    maxLen := 0
    start := 0
    left := 0
    
    for right := 0; right < len(s); right++ {
        if index, exists := charIndex[s[right]]; exists && index >= left {
            left = index + 1
        }
        
        charIndex[s[right]] = right
        
        if right-left+1 > maxLen {
            maxLen = right - left + 1
            start = left
        }
    }
    
    return s[start : start+maxLen]
}
```

### Complexity
- **Time Complexity:** O(n) for sliding window, O(n²) for brute force
- **Space Complexity:** O(min(m, n)) where m is charset size
