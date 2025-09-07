# Group Anagrams

### Problem
Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example:**
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]
```

### Golang Solution

```go
import "sort"

func groupAnagrams(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        // Sort characters to create a key
        chars := []rune(str)
        sort.Slice(chars, func(i, j int) bool {
            return chars[i] < chars[j]
        })
        key := string(chars)
        
        groups[key] = append(groups[key], str)
    }
    
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Character Count**
```go
func groupAnagramsCount(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        count := make([]int, 26)
        for _, char := range str {
            count[char-'a']++
        }
        
        // Create key from character count
        key := ""
        for i := 0; i < 26; i++ {
            key += fmt.Sprintf("%d,", count[i])
        }
        
        groups[key] = append(groups[key], str)
    }
    
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}
```

#### **Using Prime Numbers**
```go
func groupAnagramsPrime(strs []string) [][]string {
    primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101}
    groups := make(map[int][]string)
    
    for _, str := range strs {
        key := 1
        for _, char := range str {
            key *= primes[char-'a']
        }
        
        groups[key] = append(groups[key], str)
    }
    
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n × m log m) where n is number of strings, m is average length
- **Space Complexity:** O(n × m)