---
# Auto-generated front matter
Title: Firstuniquecharacterinstring
LastUpdated: 2025-11-06T20:45:58.686034
Tags: []
Status: draft
---

# First Unique Character in a String

### Problem
Given a string `s`, find the first non-repeating character in it and return its index. If it does not exist, return -1.

**Example:**
```
Input: s = "leetcode"
Output: 0

Input: s = "loveleetcode"
Output: 2

Input: s = "aabb"
Output: -1
```

### Golang Solution

```go
func firstUniqChar(s string) int {
    charCount := make(map[rune]int)
    
    // Count character frequencies
    for _, char := range s {
        charCount[char]++
    }
    
    // Find first unique character
    for i, char := range s {
        if charCount[char] == 1 {
            return i
        }
    }
    
    return -1
}
```

### Alternative Solutions

#### **Using Array for ASCII Characters**
```go
func firstUniqCharArray(s string) int {
    charCount := make([]int, 26) // For lowercase letters
    
    // Count character frequencies
    for i := 0; i < len(s); i++ {
        charCount[s[i]-'a']++
    }
    
    // Find first unique character
    for i := 0; i < len(s); i++ {
        if charCount[s[i]-'a'] == 1 {
            return i
        }
    }
    
    return -1
}
```

#### **Using Two Passes**
```go
func firstUniqCharTwoPass(s string) int {
    charCount := make(map[rune]int)
    
    // First pass: count frequencies
    for _, char := range s {
        charCount[char]++
    }
    
    // Second pass: find first unique
    for i, char := range s {
        if charCount[char] == 1 {
            return i
        }
    }
    
    return -1
}
```

#### **Using Index Tracking**
```go
func firstUniqCharIndex(s string) int {
    charIndex := make(map[rune]int)
    charCount := make(map[rune]int)
    
    for i, char := range s {
        if _, exists := charIndex[char]; !exists {
            charIndex[char] = i
        }
        charCount[char]++
    }
    
    minIndex := len(s)
    for char, count := range charCount {
        if count == 1 && charIndex[char] < minIndex {
            minIndex = charIndex[char]
        }
    }
    
    if minIndex == len(s) {
        return -1
    }
    
    return minIndex
}
```

#### **Using Set for Seen Characters**
```go
func firstUniqCharSet(s string) int {
    seen := make(map[rune]bool)
    unique := make(map[rune]int)
    
    for i, char := range s {
        if seen[char] {
            delete(unique, char)
        } else {
            seen[char] = true
            unique[char] = i
        }
    }
    
    minIndex := len(s)
    for _, index := range unique {
        if index < minIndex {
            minIndex = index
        }
    }
    
    if minIndex == len(s) {
        return -1
    }
    
    return minIndex
}
```

#### **Return Character and Index**
```go
type UniqueCharResult struct {
    Index int
    Char  rune
    Found bool
}

func firstUniqCharWithChar(s string) UniqueCharResult {
    charCount := make(map[rune]int)
    
    // Count character frequencies
    for _, char := range s {
        charCount[char]++
    }
    
    // Find first unique character
    for i, char := range s {
        if charCount[char] == 1 {
            return UniqueCharResult{
                Index: i,
                Char:  char,
                Found: true,
            }
        }
    }
    
    return UniqueCharResult{Found: false}
}
```

#### **Find All Unique Characters**
```go
func findAllUniqueChars(s string) []int {
    charCount := make(map[rune]int)
    var result []int
    
    // Count character frequencies
    for _, char := range s {
        charCount[char]++
    }
    
    // Find all unique characters
    for i, char := range s {
        if charCount[char] == 1 {
            result = append(result, i)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for constant character set, O(n) for hash map
