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

**Constraints:**
- 1 ≤ s.length, t.length ≤ 5 × 10⁴
- s and t consist of lowercase English letters only

### Explanation

#### **Hash Map Approach**
- Count frequency of each character in both strings
- Compare the frequency maps
- Time Complexity: O(n)
- Space Complexity: O(1) - limited by alphabet size

#### **Sorting Approach**
- Sort both strings and compare
- Time Complexity: O(n log n)
- Space Complexity: O(1)

### Golang Solution

```go
func isAnagram(s string, t string) bool {
    if len(s) != len(t) {
        return false
    }
    
    charCount := make(map[rune]int)
    
    // Count characters in s
    for _, char := range s {
        charCount[char]++
    }
    
    // Decrease count for characters in t
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

#### **Array Approach (for lowercase letters)**
```go
func isAnagramArray(s string, t string) bool {
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

### Notes / Variations

#### **Related Problems**
- **Group Anagrams**: Group strings that are anagrams
- **Find All Anagrams in a String**: Find all anagram occurrences
- **Permutation in String**: Check if permutation exists
- **Valid Palindrome**: Check if string is palindrome
