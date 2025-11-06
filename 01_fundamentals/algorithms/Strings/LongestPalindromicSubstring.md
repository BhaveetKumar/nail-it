---
# Auto-generated front matter
Title: Longestpalindromicsubstring
LastUpdated: 2025-11-06T20:45:58.684403
Tags: []
Status: draft
---

# Longest Palindromic Substring

### Problem
Given a string `s`, return the longest palindromic substring in `s`.

**Example:**
```
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Input: s = "cbbd"
Output: "bb"
```

### Golang Solution

```go
func longestPalindrome(s string) string {
    if len(s) == 0 {
        return ""
    }
    
    start, maxLen := 0, 1
    
    for i := 0; i < len(s); i++ {
        // Check for odd length palindromes
        len1 := expandAroundCenter(s, i, i)
        // Check for even length palindromes
        len2 := expandAroundCenter(s, i, i+1)
        
        max := len1
        if len2 > len1 {
            max = len2
        }
        
        if max > maxLen {
            maxLen = max
            start = i - (max-1)/2
        }
    }
    
    return s[start : start+maxLen]
}

func expandAroundCenter(s string, left, right int) int {
    for left >= 0 && right < len(s) && s[left] == s[right] {
        left--
        right++
    }
    return right - left - 1
}
```

### Complexity
- **Time Complexity:** O(n²)
- **Space Complexity:** O(1)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
