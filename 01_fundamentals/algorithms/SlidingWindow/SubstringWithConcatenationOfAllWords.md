# Substring with Concatenation of All Words

### Problem
You are given a string `s` and an array of strings `words`. All the strings of `words` are of the same length.

A concatenated substring in `s` is a substring that contains all the strings in any permutation of `words` concatenated.

For example, if `words = ["ab","cd","ef"]`, then `"abcdef"`, `"abefcd"`, `"cdabef"`, `"cdefab"`, `"efabcd"`, and `"efcdab"` are all concatenated strings. `"acdbef"` is not a concatenated substring because it is not the concatenation of any permutation of `words`.

Return the starting indices of all concatenated substrings in `s`. You can return the answer in any order.

**Example:**
```
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoo" and "foobar" respectively.

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
```

### Golang Solution

```go
func findSubstring(s string, words []string) []int {
    if len(words) == 0 || len(s) == 0 {
        return []int{}
    }
    
    wordLen := len(words[0])
    totalLen := len(words) * wordLen
    
    if len(s) < totalLen {
        return []int{}
    }
    
    wordCount := make(map[string]int)
    for _, word := range words {
        wordCount[word]++
    }
    
    var result []int
    
    for i := 0; i <= len(s)-totalLen; i++ {
        seen := make(map[string]int)
        j := 0
        
        for j < len(words) {
            word := s[i+j*wordLen : i+(j+1)*wordLen]
            
            if count, exists := wordCount[word]; exists {
                seen[word]++
                if seen[word] > count {
                    break
                }
            } else {
                break
            }
            j++
        }
        
        if j == len(words) {
            result = append(result, i)
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Sliding Window Approach**
```go
func findSubstringSlidingWindow(s string, words []string) []int {
    if len(words) == 0 || len(s) == 0 {
        return []int{}
    }
    
    wordLen := len(words[0])
    totalLen := len(words) * wordLen
    
    if len(s) < totalLen {
        return []int{}
    }
    
    wordCount := make(map[string]int)
    for _, word := range words {
        wordCount[word]++
    }
    
    var result []int
    
    for offset := 0; offset < wordLen; offset++ {
        left := offset
        seen := make(map[string]int)
        count := 0
        
        for right := offset; right <= len(s)-wordLen; right += wordLen {
            word := s[right : right+wordLen]
            
            if expectedCount, exists := wordCount[word]; exists {
                seen[word]++
                count++
                
                for seen[word] > expectedCount {
                    leftWord := s[left : left+wordLen]
                    seen[leftWord]--
                    count--
                    left += wordLen
                }
                
                if count == len(words) {
                    result = append(result, left)
                }
            } else {
                seen = make(map[string]int)
                count = 0
                left = right + wordLen
            }
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n × m × k) where n is string length, m is number of words, k is word length
- **Space Complexity:** O(m)
