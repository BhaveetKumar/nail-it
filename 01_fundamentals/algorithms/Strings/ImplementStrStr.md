---
# Auto-generated front matter
Title: Implementstrstr
LastUpdated: 2025-11-06T20:45:58.688298
Tags: []
Status: draft
---

# Implement strStr()

### Problem
Given two strings `needle` and `haystack`, return the index of the first occurrence of `needle` in `haystack`, or `-1` if `needle` is not part of `haystack`.

**Example:**
```
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.

Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
```

### Golang Solution

```go
func strStr(haystack string, needle string) int {
    if len(needle) == 0 {
        return 0
    }
    
    if len(haystack) < len(needle) {
        return -1
    }
    
    for i := 0; i <= len(haystack)-len(needle); i++ {
        if haystack[i:i+len(needle)] == needle {
            return i
        }
    }
    
    return -1
}
```

### Alternative Solutions

#### **Using Built-in Function**
```go
import "strings"

func strStrBuiltin(haystack string, needle string) int {
    return strings.Index(haystack, needle)
}
```

#### **Using KMP Algorithm**
```go
func strStrKMP(haystack string, needle string) int {
    if len(needle) == 0 {
        return 0
    }
    
    if len(haystack) < len(needle) {
        return -1
    }
    
    // Build failure function
    failure := buildFailureFunction(needle)
    
    i, j := 0, 0
    
    for i < len(haystack) {
        if haystack[i] == needle[j] {
            i++
            j++
            
            if j == len(needle) {
                return i - j
            }
        } else if j > 0 {
            j = failure[j-1]
        } else {
            i++
        }
    }
    
    return -1
}

func buildFailureFunction(pattern string) []int {
    failure := make([]int, len(pattern))
    j := 0
    
    for i := 1; i < len(pattern); i++ {
        for j > 0 && pattern[i] != pattern[j] {
            j = failure[j-1]
        }
        
        if pattern[i] == pattern[j] {
            j++
        }
        
        failure[i] = j
    }
    
    return failure
}
```

#### **Using Rabin-Karp Algorithm**
```go
func strStrRabinKarp(haystack string, needle string) int {
    if len(needle) == 0 {
        return 0
    }
    
    if len(haystack) < len(needle) {
        return -1
    }
    
    const base = 256
    const mod = 1000000007
    
    // Calculate hash of needle
    needleHash := 0
    for i := 0; i < len(needle); i++ {
        needleHash = (needleHash*base + int(needle[i])) % mod
    }
    
    // Calculate hash of first window in haystack
    haystackHash := 0
    for i := 0; i < len(needle); i++ {
        haystackHash = (haystackHash*base + int(haystack[i])) % mod
    }
    
    // Calculate base^(len(needle)-1) for rolling hash
    power := 1
    for i := 0; i < len(needle)-1; i++ {
        power = (power * base) % mod
    }
    
    // Check first window
    if haystackHash == needleHash && haystack[:len(needle)] == needle {
        return 0
    }
    
    // Roll the hash
    for i := len(needle); i < len(haystack); i++ {
        // Remove leftmost character
        haystackHash = (haystackHash - int(haystack[i-len(needle)])*power) % mod
        if haystackHash < 0 {
            haystackHash += mod
        }
        
        // Add new character
        haystackHash = (haystackHash*base + int(haystack[i])) % mod
        
        // Check if hash matches
        if haystackHash == needleHash && haystack[i-len(needle)+1:i+1] == needle {
            return i - len(needle) + 1
        }
    }
    
    return -1
}
```

#### **Return All Occurrences**
```go
func strStrAll(haystack string, needle string) []int {
    if len(needle) == 0 {
        return []int{}
    }
    
    if len(haystack) < len(needle) {
        return []int{}
    }
    
    var result []int
    
    for i := 0; i <= len(haystack)-len(needle); i++ {
        if haystack[i:i+len(needle)] == needle {
            result = append(result, i)
        }
    }
    
    return result
}
```

#### **Case Insensitive Search**
```go
func strStrCaseInsensitive(haystack string, needle string) int {
    if len(needle) == 0 {
        return 0
    }
    
    if len(haystack) < len(needle) {
        return -1
    }
    
    haystackLower := strings.ToLower(haystack)
    needleLower := strings.ToLower(needle)
    
    for i := 0; i <= len(haystackLower)-len(needleLower); i++ {
        if haystackLower[i:i+len(needleLower)] == needleLower {
            return i
        }
    }
    
    return -1
}
```

### Complexity
- **Time Complexity:** O(n×m) for naive, O(n+m) for KMP/Rabin-Karp
- **Space Complexity:** O(1) for naive, O(m) for KMP