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
        
        // Check if sChar is already mapped
        if mappedT, exists := sToT[sChar]; exists {
            if mappedT != tChar {
                return false
            }
        } else {
            sToT[sChar] = tChar
        }
        
        // Check if tChar is already mapped
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

#### **Using Single Map with Set**
```go
func isIsomorphicSingleMap(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    charMap := make(map[byte]byte)
    usedChars := make(map[byte]bool)
    
    for i := 0; i < len(s); i++ {
        sChar := s[i]
        tChar := t[i]
        
        if mappedChar, exists := charMap[sChar]; exists {
            if mappedChar != tChar {
                return false
            }
        } else {
            if usedChars[tChar] {
                return false
            }
            charMap[sChar] = tChar
            usedChars[tChar] = true
        }
    }
    
    return true
}
```

#### **Using Array for ASCII Characters**
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

#### **Using String Transformation**
```go
func isIsomorphicTransform(s string, t string) bool {
    return transformString(s) == transformString(t)
}

func transformString(s string) string {
    charMap := make(map[byte]int)
    result := make([]byte, len(s))
    nextChar := 0
    
    for i := 0; i < len(s); i++ {
        if _, exists := charMap[s[i]]; !exists {
            charMap[s[i]] = nextChar
            nextChar++
        }
        result[i] = byte(charMap[s[i]] + 'a')
    }
    
    return string(result)
}
```

#### **Using Index Mapping**
```go
func isIsomorphicIndex(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    sIndices := make(map[byte][]int)
    tIndices := make(map[byte][]int)
    
    for i := 0; i < len(s); i++ {
        sIndices[s[i]] = append(sIndices[s[i]], i)
        tIndices[t[i]] = append(tIndices[t[i]], i)
    }
    
    if len(sIndices) != len(tIndices) {
        return false
    }
    
    // Check if the pattern of indices matches
    sPattern := make([]int, len(s))
    tPattern := make([]int, len(t))
    
    for i, char := range s {
        sPattern[i] = sIndices[byte(char)][0]
    }
    
    for i, char := range t {
        tPattern[i] = tIndices[byte(char)][0]
    }
    
    for i := 0; i < len(sPattern); i++ {
        if sPattern[i] != tPattern[i] {
            return false
        }
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for array, O(n) for hash map
