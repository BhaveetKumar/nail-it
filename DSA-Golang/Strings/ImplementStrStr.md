# Implement strStr()

### Problem
Given two strings `needle` and `haystack`, return the index of the first occurrence of `needle` in `haystack`, or `-1` if `needle` is not part of `haystack`.

**Example:**
```
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6. The first occurrence is at index 0.

Input: haystack = "leetcode", needle = "leeto"
Output: -1
```

### Golang Solution

```go
func strStr(haystack string, needle string) int {
    if len(needle) == 0 {
        return 0
    }
    
    if len(needle) > len(haystack) {
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

### Complexity
- **Time Complexity:** O((n-m+1) × m) where n is haystack length, m is needle length
- **Space Complexity:** O(1)
