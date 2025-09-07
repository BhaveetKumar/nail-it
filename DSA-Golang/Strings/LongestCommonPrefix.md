# Longest Common Prefix

### Problem
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string `""`.

**Example:**
```
Input: strs = ["flower","flow","flight"]
Output: "fl"

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
```

### Golang Solution

```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    prefix := strs[0]
    
    for i := 1; i < len(strs); i++ {
        for len(prefix) > 0 && !strings.HasPrefix(strs[i], prefix) {
            prefix = prefix[:len(prefix)-1]
        }
        
        if prefix == "" {
            return ""
        }
    }
    
    return prefix
}
```

### Alternative Solutions

#### **Vertical Scanning**
```go
func longestCommonPrefixVertical(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    for i := 0; i < len(strs[0]); i++ {
        char := strs[0][i]
        
        for j := 1; j < len(strs); j++ {
            if i >= len(strs[j]) || strs[j][i] != char {
                return strs[0][:i]
            }
        }
    }
    
    return strs[0]
}
```

### Complexity
- **Time Complexity:** O(S) where S is sum of all characters
- **Space Complexity:** O(1)
