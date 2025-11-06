---
# Auto-generated front matter
Title: Validpalindromeii
LastUpdated: 2025-11-06T20:45:58.699227
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

### Alternative Solutions

#### **Recursive Approach**
```go
func validPalindromeRecursive(s string) bool {
    return validPalindromeHelper(s, 0, len(s)-1, 1)
}

func validPalindromeHelper(s string, left, right, deletions int) bool {
    if left >= right {
        return true
    }
    
    if s[left] == s[right] {
        return validPalindromeHelper(s, left+1, right-1, deletions)
    }
    
    if deletions == 0 {
        return false
    }
    
    return validPalindromeHelper(s, left+1, right, deletions-1) ||
           validPalindromeHelper(s, left, right-1, deletions-1)
}
```

#### **Using Two Helper Functions**
```go
func validPalindromeTwoHelpers(s string) bool {
    left, right := 0, len(s)-1
    
    for left < right {
        if s[left] != s[right] {
            return checkPalindrome(s, left+1, right) || checkPalindrome(s, left, right-1)
        }
        left++
        right--
    }
    
    return true
}

func checkPalindrome(s string, start, end int) bool {
    for start < end {
        if s[start] != s[end] {
            return false
        }
        start++
        end--
    }
    return true
}
```

#### **Iterative with Skip Logic**
```go
func validPalindromeIterative(s string) bool {
    left, right := 0, len(s)-1
    skipped := false
    
    for left < right {
        if s[left] != s[right] {
            if skipped {
                return false
            }
            
            // Try skipping left character
            if s[left+1] == s[right] {
                left++
                skipped = true
            } else if s[left] == s[right-1] {
                right--
                skipped = true
            } else {
                return false
            }
        } else {
            left++
            right--
        }
    }
    
    return true
}
```

#### **Return All Valid Deletions**
```go
func validPalindromeWithDeletions(s string) (bool, []int) {
    var deletions []int
    
    if isPalindromeComplete(s) {
        return true, deletions
    }
    
    for i := 0; i < len(s); i++ {
        newS := s[:i] + s[i+1:]
        if isPalindromeComplete(newS) {
            deletions = append(deletions, i)
        }
    }
    
    return len(deletions) > 0, deletions
}

func isPalindromeComplete(s string) bool {
    left, right := 0, len(s)-1
    
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
- **Space Complexity:** O(1) for iterative, O(n) for recursive