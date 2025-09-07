# Find All Anagrams in a String

### Problem
Given two strings `s` and `p`, return an array of all the start indices of `p`'s anagrams in `s`. You may return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example:**
```
Input: s = "cbaebabacd", p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".

Input: s = "abab", p = "ab"
Output: [0,1,2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
```

### Golang Solution

```go
func findAnagrams(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pCount := make([]int, 26)
    sCount := make([]int, 26)
    
    // Count characters in pattern
    for _, char := range p {
        pCount[char-'a']++
    }
    
    // Initialize sliding window
    for i := 0; i < len(p); i++ {
        sCount[s[i]-'a']++
    }
    
    // Check first window
    if isEqual(pCount, sCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]-'a']--
        // Add new character
        sCount[s[i]-'a']++
        
        // Check if current window is an anagram
        if isEqual(pCount, sCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func isEqual(a, b []int) bool {
    for i := 0; i < 26; i++ {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

### Alternative Solutions

#### **Using Hash Map**
```go
func findAnagramsHashMap(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pCount := make(map[byte]int)
    sCount := make(map[byte]int)
    
    // Count characters in pattern
    for i := 0; i < len(p); i++ {
        pCount[p[i]]++
    }
    
    // Initialize sliding window
    for i := 0; i < len(p); i++ {
        sCount[s[i]]++
    }
    
    // Check first window
    if isEqualMap(pCount, sCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]]--
        if sCount[s[i-len(p)]] == 0 {
            delete(sCount, s[i-len(p)])
        }
        
        // Add new character
        sCount[s[i]]++
        
        // Check if current window is an anagram
        if isEqualMap(pCount, sCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func isEqualMap(a, b map[byte]int) bool {
    if len(a) != len(b) {
        return false
    }
    
    for key, value := range a {
        if b[key] != value {
            return false
        }
    }
    
    return true
}
```

#### **Optimized Array Approach**
```go
func findAnagramsOptimized(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pCount := make([]int, 26)
    sCount := make([]int, 26)
    
    // Count characters in pattern
    for _, char := range p {
        pCount[char-'a']++
    }
    
    // Initialize sliding window
    for i := 0; i < len(p); i++ {
        sCount[s[i]-'a']++
    }
    
    // Check first window
    if isEqualOptimized(pCount, sCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]-'a']--
        // Add new character
        sCount[s[i]-'a']++
        
        // Check if current window is an anagram
        if isEqualOptimized(pCount, sCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func isEqualOptimized(a, b []int) bool {
    for i := 0; i < 26; i++ {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

#### **Using Sort**
```go
import "sort"

func findAnagramsSort(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pSorted := sortString(p)
    
    for i := 0; i <= len(s)-len(p); i++ {
        substring := s[i : i+len(p)]
        if sortString(substring) == pSorted {
            result = append(result, i)
        }
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
- **Time Complexity:** O(n) for sliding window, O(n × m log m) for sorting
- **Space Complexity:** O(1) for array, O(m) for hash map
