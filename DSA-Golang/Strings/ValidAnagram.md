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
    
    charCount := make(map[rune]int)
    
    for _, char := range s {
        charCount[char]++
    }
    
    for _, char := range t {
        charCount[char]--
        if charCount[char] < 0 {
            return false
        }
    }
    
    return true
}
```

### Alternative Solutions

#### **Using Array for ASCII Characters**
```go
func isAnagramArray(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    charCount := make([]int, 26) // For lowercase letters
    
    for i := 0; i < len(s); i++ {
        charCount[s[i]-'a']++
        charCount[t[i]-'a']--
    }
    
    for _, count := range charCount {
        if count != 0 {
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

#### **Using Two Maps**
```go
func isAnagramTwoMaps(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    sCount := make(map[rune]int)
    tCount := make(map[rune]int)
    
    for _, char := range s {
        sCount[char]++
    }
    
    for _, char := range t {
        tCount[char]++
    }
    
    if len(sCount) != len(tCount) {
        return false
    }
    
    for char, count := range sCount {
        if tCount[char] != count {
            return false
        }
    }
    
    return true
}
```

#### **Using XOR (Limited Use Case)**
```go
func isAnagramXOR(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    xor := 0
    sum := 0
    
    for i := 0; i < len(s); i++ {
        xor ^= int(s[i])
        xor ^= int(t[i])
        sum += int(s[i])
        sum -= int(t[i])
    }
    
    return xor == 0 && sum == 0
}
```

### Complexity
- **Time Complexity:** O(n) for hash map, O(n log n) for sorting
- **Space Complexity:** O(1) for array, O(n) for hash map