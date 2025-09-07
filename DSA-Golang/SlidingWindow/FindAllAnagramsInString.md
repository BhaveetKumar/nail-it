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
```

### Golang Solution

```go
func findAnagrams(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pCount := make(map[byte]int)
    windowCount := make(map[byte]int)
    
    // Count characters in pattern
    for i := 0; i < len(p); i++ {
        pCount[p[i]]++
    }
    
    // Initialize window
    for i := 0; i < len(p); i++ {
        windowCount[s[i]]++
    }
    
    if mapsEqual(pCount, windowCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Add new character
        windowCount[s[i]]++
        
        // Remove old character
        windowCount[s[i-len(p)]]--
        if windowCount[s[i-len(p)]] == 0 {
            delete(windowCount, s[i-len(p)])
        }
        
        // Check if current window is an anagram
        if mapsEqual(pCount, windowCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func mapsEqual(map1, map2 map[byte]int) bool {
    if len(map1) != len(map2) {
        return false
    }
    
    for k, v := range map1 {
        if map2[k] != v {
            return false
        }
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) - limited by alphabet size
