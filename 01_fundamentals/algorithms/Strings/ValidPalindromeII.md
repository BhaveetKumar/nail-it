---
# Auto-generated front matter
Title: Validpalindromeii
LastUpdated: 2025-11-06T20:45:58.685555
Tags: []
Status: draft
---

# Valid Palindrome II

### Problem
Given a string `s`, return `true` if the `s` can be palindrome after deleting at most one character from it.

**Example:**
```
Input: s = "aba"
Output: true

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.

Input: s = "abc"
Output: false
```

### Golang Solution

```go
func validPalindrome(s string) bool {
    left, right := 0, len(s)-1
    
    for left < right {
        if s[left] != s[right] {
            return isPalindrome(s, left+1, right) || isPalindrome(s, left, right-1)
        }
        left++
        right--
    }
    
    return true
}

func isPalindrome(s string, left, right int) bool {
    for left < right {
        if s[left] != s[right] {
            return false
        }
        left++
        right--
    }
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
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


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
