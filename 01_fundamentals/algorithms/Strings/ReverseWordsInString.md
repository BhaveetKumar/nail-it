---
# Auto-generated front matter
Title: Reversewordsinstring
LastUpdated: 2025-11-06T20:45:58.689402
Tags: []
Status: draft
---

# Reverse Words in a String

### Problem
Given an input string `s`, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in `s` will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that `s` may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

**Example:**
```
Input: s = "the sky is blue"
Output: "blue is sky the"

Input: s = "  hello world  "
Output: "world hello"

Input: s = "a good   example"
Output: "example good a"
```

### Golang Solution

```go
import "strings"

func reverseWords(s string) string {
    // Split by spaces and filter out empty strings
    words := strings.Fields(s)
    
    // Reverse the slice
    for i, j := 0, len(words)-1; i < j; i, j = i+1, j-1 {
        words[i], words[j] = words[j], words[i]
    }
    
    // Join with single space
    return strings.Join(words, " ")
}
```

### Alternative Solutions

#### **Manual Implementation**
```go
func reverseWordsManual(s string) string {
    // Trim leading and trailing spaces
    s = strings.TrimSpace(s)
    
    // Split into words
    var words []string
    var current strings.Builder
    
    for _, char := range s {
        if char == ' ' {
            if current.Len() > 0 {
                words = append(words, current.String())
                current.Reset()
            }
        } else {
            current.WriteRune(char)
        }
    }
    
    if current.Len() > 0 {
        words = append(words, current.String())
    }
    
    // Reverse words
    for i, j := 0, len(words)-1; i < j; i, j = i+1, j-1 {
        words[i], words[j] = words[j], words[i]
    }
    
    return strings.Join(words, " ")
}
```

#### **In-Place Reverse**
```go
func reverseWordsInPlace(s string) string {
    // Convert to byte slice for in-place modification
    bytes := []byte(s)
    
    // Reverse the entire string
    reverseBytes(bytes, 0, len(bytes)-1)
    
    // Reverse each word
    start := 0
    for i := 0; i <= len(bytes); i++ {
        if i == len(bytes) || bytes[i] == ' ' {
            reverseBytes(bytes, start, i-1)
            start = i + 1
        }
    }
    
    // Clean up multiple spaces
    return cleanSpaces(string(bytes))
}

func reverseBytes(bytes []byte, start, end int) {
    for start < end {
        bytes[start], bytes[end] = bytes[end], bytes[start]
        start++
        end--
    }
}

func cleanSpaces(s string) string {
    var result strings.Builder
    inWord := false
    
    for _, char := range s {
        if char != ' ' {
            result.WriteRune(char)
            inWord = true
        } else if inWord {
            result.WriteRune(' ')
            inWord = false
        }
    }
    
    return strings.TrimSpace(result.String())
}
```

#### **Stack Approach**
```go
func reverseWordsStack(s string) string {
    var stack []string
    var current strings.Builder
    
    for _, char := range s {
        if char == ' ' {
            if current.Len() > 0 {
                stack = append(stack, current.String())
                current.Reset()
            }
        } else {
            current.WriteRune(char)
        }
    }
    
    if current.Len() > 0 {
        stack = append(stack, current.String())
    }
    
    var result strings.Builder
    for i := len(stack) - 1; i >= 0; i-- {
        if result.Len() > 0 {
            result.WriteString(" ")
        }
        result.WriteString(stack[i])
    }
    
    return result.String()
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
