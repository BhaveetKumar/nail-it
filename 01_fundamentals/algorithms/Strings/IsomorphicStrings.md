---
# Auto-generated front matter
Title: Isomorphicstrings
LastUpdated: 2025-11-06T20:45:58.688798
Tags: []
Status: draft
---

# Isomorphic Strings

### Problem
Given two strings `s` and `t`, determine if they are isomorphic.

Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t`.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

**Example:**
```
Input: s = "egg", t = "add"
Output: true

Input: s = "foo", t = "bar"
Output: false

Input: s = "paper", t = "title"
Output: true
```

### Golang Solution

```go
func isIsomorphic(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    sToT := make(map[byte]byte)
    tToS := make(map[byte]byte)
    
    for i := 0; i < len(s); i++ {
        sChar := s[i]
        tChar := t[i]
        
        if mappedT, exists := sToT[sChar]; exists {
            if mappedT != tChar {
                return false
            }
        } else {
            sToT[sChar] = tChar
        }
        
        if mappedS, exists := tToS[tChar]; exists {
            if mappedS != sChar {
                return false
            }
        } else {
            tToS[tChar] = sChar
        }
    }
    
    return true
}
```

### Alternative Solutions

#### **Using Single Map**
```go
func isIsomorphicSingleMap(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    mapping := make(map[byte]byte)
    used := make(map[byte]bool)
    
    for i := 0; i < len(s); i++ {
        sChar := s[i]
        tChar := t[i]
        
        if mapped, exists := mapping[sChar]; exists {
            if mapped != tChar {
                return false
            }
        } else {
            if used[tChar] {
                return false
            }
            mapping[sChar] = tChar
            used[tChar] = true
        }
    }
    
    return true
}
```

#### **Using Array**
```go
func isIsomorphicArray(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    sToT := make([]byte, 256)
    tToS := make([]byte, 256)
    
    for i := 0; i < len(s); i++ {
        sChar := s[i]
        tChar := t[i]
        
        if sToT[sChar] != 0 && sToT[sChar] != tChar {
            return false
        }
        if tToS[tChar] != 0 && tToS[tChar] != sChar {
            return false
        }
        
        sToT[sChar] = tChar
        tToS[tChar] = sChar
    }
    
    return true
}
```

#### **Return Mapping**
```go
type IsomorphicResult struct {
    IsIsomorphic bool
    Mapping      map[byte]byte
    ReverseMapping map[byte]byte
}

func isIsomorphicWithMapping(s string, t string) IsomorphicResult {
    if len(s) != len(t) {
        return IsomorphicResult{
            IsIsomorphic: false,
            Mapping:      make(map[byte]byte),
            ReverseMapping: make(map[byte]byte),
        }
    }
    
    sToT := make(map[byte]byte)
    tToS := make(map[byte]byte)
    
    for i := 0; i < len(s); i++ {
        sChar := s[i]
        tChar := t[i]
        
        if mappedT, exists := sToT[sChar]; exists {
            if mappedT != tChar {
                return IsomorphicResult{
                    IsIsomorphic: false,
                    Mapping:      sToT,
                    ReverseMapping: tToS,
                }
            }
        } else {
            sToT[sChar] = tChar
        }
        
        if mappedS, exists := tToS[tChar]; exists {
            if mappedS != sChar {
                return IsomorphicResult{
                    IsIsomorphic: false,
                    Mapping:      sToT,
                    ReverseMapping: tToS,
                }
            }
        } else {
            tToS[tChar] = sChar
        }
    }
    
    return IsomorphicResult{
        IsIsomorphic: true,
        Mapping:      sToT,
        ReverseMapping: tToS,
    }
}
```

#### **Transform String**
```go
func transformString(s string, mapping map[byte]byte) string {
    result := make([]byte, len(s))
    
    for i := 0; i < len(s); i++ {
        if mapped, exists := mapping[s[i]]; exists {
            result[i] = mapped
        } else {
            result[i] = s[i]
        }
    }
    
    return string(result)
}
```

#### **Find All Isomorphic Strings**
```go
func findAllIsomorphic(s string, wordList []string) []string {
    var isomorphic []string
    
    for _, word := range wordList {
        if isIsomorphic(s, word) {
            isomorphic = append(isomorphic, word)
        }
    }
    
    return isomorphic
}
```

#### **Return Isomorphism Groups**
```go
func groupIsomorphicStrings(words []string) [][]string {
    groups := make(map[string][]string)
    
    for _, word := range words {
        key := getIsomorphismKey(word)
        groups[key] = append(groups[key], word)
    }
    
    var result [][]string
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

func getIsomorphismKey(s string) string {
    mapping := make(map[byte]int)
    key := make([]byte, len(s))
    nextId := 0
    
    for i := 0; i < len(s); i++ {
        if id, exists := mapping[s[i]]; exists {
            key[i] = byte(id)
        } else {
            mapping[s[i]] = nextId
            key[i] = byte(nextId)
            nextId++
        }
    }
    
    return string(key)
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the strings
- **Space Complexity:** O(1) for fixed character set, O(n) for hash maps