# Longest Substring with At Most K Distinct Characters

### Problem
Given a string `s` and an integer `k`, return the length of the longest substring of `s` that contains at most `k` distinct characters.

**Example:**
```
Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
```

### Golang Solution

```go
func lengthOfLongestSubstringKDistinct(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        // Shrink window if we have more than k distinct characters
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
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
func lengthOfLongestSubstringKDistinctArray(s string, k int) int {
    if k == 0 {
        return 0
    }
    
    charCount := make([]int, 128) // ASCII characters
    distinctCount := 0
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        if charCount[s[right]] == 0 {
            distinctCount++
        }
        charCount[s[right]]++
        
        for distinctCount > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                distinctCount--
            }
            left++
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

#### **Return the Actual Substring**
```go
func longestSubstringKDistinct(s string, k int) string {
    if k == 0 {
        return ""
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    start := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }
        
        if right-left+1 > maxLen {
            maxLen = right - left + 1
            start = left
        }
    }
    
    return s[start : start+maxLen]
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(k)
