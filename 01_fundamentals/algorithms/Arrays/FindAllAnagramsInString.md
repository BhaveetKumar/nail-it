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
    
    // Check if first window is anagram
    if isEqual(pCount, sCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]-'a']--
        // Add new character
        sCount[s[i]-'a']++
        
        // Check if current window is anagram
        if isEqual(pCount, sCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func isEqual(arr1, arr2 []int) bool {
    for i := 0; i < 26; i++ {
        if arr1[i] != arr2[i] {
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
    pCount := make(map[rune]int)
    sCount := make(map[rune]int)
    
    // Count characters in pattern
    for _, char := range p {
        pCount[char]++
    }
    
    // Initialize sliding window
    for i := 0; i < len(p); i++ {
        sCount[rune(s[i])]++
    }
    
    // Check if first window is anagram
    if isEqualMap(pCount, sCount) {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        leftChar := rune(s[i-len(p)])
        sCount[leftChar]--
        if sCount[leftChar] == 0 {
            delete(sCount, leftChar)
        }
        
        // Add new character
        rightChar := rune(s[i])
        sCount[rightChar]++
        
        // Check if current window is anagram
        if isEqualMap(pCount, sCount) {
            result = append(result, i-len(p)+1)
        }
    }
    
    return result
}

func isEqualMap(map1, map2 map[rune]int) bool {
    if len(map1) != len(map2) {
        return false
    }
    
    for key, value := range map1 {
        if map2[key] != value {
            return false
        }
    }
    
    return true
}
```

#### **Using Sorting**
```go
import "sort"

func findAnagramsSort(s string, p string) []int {
    if len(s) < len(p) {
        return []int{}
    }
    
    var result []int
    pSorted := sortString(p)
    
    for i := 0; i <= len(s)-len(p); i++ {
        window := s[i : i+len(p)]
        if sortString(window) == pSorted {
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

#### **Return with Substrings**
```go
type AnagramResult struct {
    Index     int
    Substring string
}

func findAnagramsWithSubstrings(s string, p string) []AnagramResult {
    if len(s) < len(p) {
        return []AnagramResult{}
    }
    
    var result []AnagramResult
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
    
    // Check if first window is anagram
    if isEqual(pCount, sCount) {
        result = append(result, AnagramResult{
            Index:     0,
            Substring: s[0:len(p)],
        })
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]-'a']--
        // Add new character
        sCount[s[i]-'a']++
        
        // Check if current window is anagram
        if isEqual(pCount, sCount) {
            result = append(result, AnagramResult{
                Index:     i - len(p) + 1,
                Substring: s[i-len(p)+1 : i+1],
            })
        }
    }
    
    return result
}
```

#### **Count All Anagrams**
```go
func countAnagrams(s string, p string) int {
    if len(s) < len(p) {
        return 0
    }
    
    count := 0
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
    
    // Check if first window is anagram
    if isEqual(pCount, sCount) {
        count++
    }
    
    // Slide the window
    for i := len(p); i < len(s); i++ {
        // Remove leftmost character
        sCount[s[i-len(p)]-'a']--
        // Add new character
        sCount[s[i]-'a']++
        
        // Check if current window is anagram
        if isEqual(pCount, sCount) {
            count++
        }
    }
    
    return count
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of string s
- **Space Complexity:** O(1) for fixed character set, O(k) for hash map where k is unique characters
