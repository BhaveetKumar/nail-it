---
# Auto-generated front matter
Title: Longestsubstringwithatmosttwodistinctcharacters
LastUpdated: 2025-11-06T20:45:58.712028
Tags: []
Status: draft
---

# Longest Substring with At Most Two Distinct Characters

### Problem
Given a string `s`, return the length of the longest substring that contains at most two distinct characters.

**Example:**
```
Input: s = "eceba"
Output: 3
Explanation: The substring is "ece" which its length is 3.

Input: s = "ccaabbb"
Output: 5
Explanation: The substring is "aabbb" which its length is 5.
```

### Golang Solution

```go
func lengthOfLongestSubstringTwoDistinct(s string) int {
    if len(s) <= 2 {
        return len(s)
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        for len(charCount) > 2 {
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
func lengthOfLongestSubstringTwoDistinctArray(s string) int {
    if len(s) <= 2 {
        return len(s)
    }
    
    charCount := make([]int, 128)
    distinctCount := 0
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        if charCount[s[right]] == 0 {
            distinctCount++
        }
        charCount[s[right]]++
        
        for distinctCount > 2 {
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
func longestSubstringTwoDistinct(s string) string {
    if len(s) <= 2 {
        return s
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    start := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
        for len(charCount) > 2 {
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

#### **Using Two Pointers with Set**
```go
func lengthOfLongestSubstringTwoDistinctSet(s string) int {
    if len(s) <= 2 {
        return len(s)
    }
    
    charSet := make(map[byte]bool)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        charSet[s[right]] = true
        
        for len(charSet) > 2 {
            // Find the leftmost character to remove
            tempLeft := left
            for tempLeft < right {
                if s[tempLeft] != s[right] && s[tempLeft] != s[right-1] {
                    delete(charSet, s[tempLeft])
                    left = tempLeft + 1
                    break
                }
                tempLeft++
            }
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

#### **Generalized for K Distinct Characters**
```go
func lengthOfLongestSubstringKDistinct(s string, k int) int {
    if len(s) <= k {
        return len(s)
    }
    
    charCount := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        charCount[s[right]]++
        
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
```

#### **Using Sliding Window with Last Seen**
```go
func lengthOfLongestSubstringTwoDistinctLastSeen(s string) int {
    if len(s) <= 2 {
        return len(s)
    }
    
    lastSeen := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        lastSeen[s[right]] = right
        
        if len(lastSeen) > 2 {
            // Find the character with the smallest last seen index
            minIndex := right
            var charToRemove byte
            
            for char, index := range lastSeen {
                if index < minIndex {
                    minIndex = index
                    charToRemove = char
                }
            }
            
            delete(lastSeen, charToRemove)
            left = minIndex + 1
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for constant character set
