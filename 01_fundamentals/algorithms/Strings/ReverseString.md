---
# Auto-generated front matter
Title: Reversestring
LastUpdated: 2025-11-06T20:45:58.686668
Tags: []
Status: draft
---

# Reverse String

### Problem
Write a function that reverses a string. The input string is given as an array of characters `s`.

You must do this by modifying the input array in-place with O(1) extra memory.

**Example:**
```
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
```

### Golang Solution

```go
func reverseString(s []byte) {
    left, right := 0, len(s)-1
    
    for left < right {
        s[left], s[right] = s[right], s[left]
        left++
        right--
    }
}
```

### Alternative Solutions

#### **Using Recursion**
```go
func reverseStringRecursive(s []byte) {
    reverseHelper(s, 0, len(s)-1)
}

func reverseHelper(s []byte, left, right int) {
    if left >= right {
        return
    }
    
    s[left], s[right] = s[right], s[left]
    reverseHelper(s, left+1, right-1)
}
```

#### **Using Stack**
```go
func reverseStringStack(s []byte) {
    stack := make([]byte, len(s))
    
    // Push all characters to stack
    for i, char := range s {
        stack[i] = char
    }
    
    // Pop from stack to reverse
    for i := 0; i < len(s); i++ {
        s[i] = stack[len(s)-1-i]
    }
}
```

#### **Using Built-in Functions**
```go
import "strings"

func reverseStringBuiltin(s []byte) {
    str := string(s)
    reversed := reverseStringHelper(str)
    copy(s, []byte(reversed))
}

func reverseStringHelper(s string) string {
    runes := []rune(s)
    for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
        runes[i], runes[j] = runes[j], runes[i]
    }
    return string(runes)
}
```

#### **Using XOR**
```go
func reverseStringXOR(s []byte) {
    left, right := 0, len(s)-1
    
    for left < right {
        s[left] ^= s[right]
        s[right] ^= s[left]
        s[left] ^= s[right]
        left++
        right--
    }
}
```

#### **Return Reversed String**
```go
func reverseStringReturn(s []byte) string {
    left, right := 0, len(s)-1
    
    for left < right {
        s[left], s[right] = s[right], s[left]
        left++
        right--
    }
    
    return string(s)
}
```

#### **Reverse Words in String**
```go
func reverseWords(s []byte) {
    // First reverse the entire string
    reverseString(s)
    
    // Then reverse each word
    start := 0
    for i := 0; i <= len(s); i++ {
        if i == len(s) || s[i] == ' ' {
            reverseWord(s, start, i-1)
            start = i + 1
        }
    }
}

func reverseWord(s []byte, start, end int) {
    for start < end {
        s[start], s[end] = s[end], s[start]
        start++
        end--
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for in-place, O(n) for stack/recursion