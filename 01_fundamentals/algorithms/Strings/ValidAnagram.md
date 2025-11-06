---
# Auto-generated front matter
Title: Validanagram
LastUpdated: 2025-11-06T20:45:58.684600
Tags: []
Status: draft
---

# Valid Anagram

### Problem
Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example:**
```
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```

### Golang Solution

```go
func isAnagram(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    count := make([]int, 26)
    
    for i := 0; i < len(s); i++ {
        count[s[i]-'a']++
        count[t[i]-'a']--
    }
    
    for _, c := range count {
        if c != 0 {
            return false
        }
    }
    
    return true
}
```

### Alternative Solutions

#### **Using Hash Map**
```go
func isAnagramHashMap(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    count := make(map[rune]int)
    
    for _, char := range s {
        count[char]++
    }
    
    for _, char := range t {
        count[char]--
        if count[char] < 0 {
            return false
        }
    }
    
    return true
}
```

#### **Using Sorting**
```go
import "sort"

func isAnagramSort(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    sRunes := []rune(s)
    tRunes := []rune(t)
    
    sort.Slice(sRunes, func(i, j int) bool {
        return sRunes[i] < sRunes[j]
    })
    
    sort.Slice(tRunes, func(i, j int) bool {
        return tRunes[i] < tRunes[j]
    })
    
    return string(sRunes) == string(tRunes)
}
```

#### **Return Character Differences**
```go
type AnagramResult struct {
    IsAnagram bool
    Differences map[rune]int
    MissingChars []rune
    ExtraChars   []rune
}

func isAnagramWithDetails(s string, t string) AnagramResult {
    if len(s) != len(t) {
        return AnagramResult{
            IsAnagram: false,
            Differences: make(map[rune]int),
        }
    }
    
    count := make(map[rune]int)
    
    for _, char := range s {
        count[char]++
    }
    
    for _, char := range t {
        count[char]--
    }
    
    var missingChars, extraChars []rune
    differences := make(map[rune]int)
    
    for char, diff := range count {
        if diff != 0 {
            differences[char] = diff
            if diff > 0 {
                missingChars = append(missingChars, char)
            } else {
                extraChars = append(extraChars, char)
            }
        }
    }
    
    return AnagramResult{
        IsAnagram:    len(differences) == 0,
        Differences:  differences,
        MissingChars: missingChars,
        ExtraChars:   extraChars,
    }
}
```

#### **Case Insensitive**
```go
import "strings"

func isAnagramCaseInsensitive(s string, t string) bool {
    s = strings.ToLower(s)
    t = strings.ToLower(t)
    return isAnagram(s, t)
}
```

#### **Return All Anagrams**
```go
func findAllAnagrams(s string, wordList []string) []string {
    var anagrams []string
    
    for _, word := range wordList {
        if isAnagram(s, word) {
            anagrams = append(anagrams, word)
        }
    }
    
    return anagrams
}
```

#### **Return Anagram Groups**
```go
func groupAnagrams(words []string) [][]string {
    groups := make(map[string][]string)
    
    for _, word := range words {
        key := sortString(word)
        groups[key] = append(groups[key], word)
    }
    
    var result [][]string
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

func sortString(s string) string {
    runes := []rune(s)
    sort.Slice(runes, func(i, j int) bool {
        return runes[i] < runes[j]
    })
    return string(runes)
}
```

### Complexity
- **Time Complexity:** O(n) for counting, O(n log n) for sorting
- **Space Complexity:** O(1) for fixed character set, O(n) for hash map