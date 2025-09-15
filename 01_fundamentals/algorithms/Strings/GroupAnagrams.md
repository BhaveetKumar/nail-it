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
        // Sort characters to create key
        sorted := sortString(str)
        groups[sorted] = append(groups[sorted], str)
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

### Alternative Solutions

#### **Using Character Count**
```go
func groupAnagramsCount(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        // Create key from character count
        key := createCountKey(str)
        groups[key] = append(groups[key], str)
    }
    
    var result [][]string
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

func createCountKey(s string) string {
    count := make([]int, 26)
    
    for _, char := range s {
        count[char-'a']++
    }
    
    var key strings.Builder
    for i := 0; i < 26; i++ {
        key.WriteString(fmt.Sprintf("%d,", count[i]))
    }
    
    return key.String()
}
```

#### **Using Hash of Character Count**
```go
import "fmt"

func groupAnagramsHash(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        // Create hash from character count
        key := createHashKey(str)
        groups[key] = append(groups[key], str)
    }
    
    var result [][]string
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

func createHashKey(s string) string {
    count := make([]int, 26)
    
    for _, char := range s {
        count[char-'a']++
    }
    
    return fmt.Sprintf("%v", count)
}
```

#### **Using Prime Numbers**
```go
func groupAnagramsPrime(strs []string) [][]string {
    // Map each letter to a prime number
    primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101}
    
    groups := make(map[int][]string)
    
    for _, str := range strs {
        // Calculate product of prime numbers
        key := 1
        for _, char := range str {
            key *= primes[char-'a']
        }
        groups[key] = append(groups[key], str)
    }
    
    var result [][]string
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}
```

#### **Return with Counts**
```go
type AnagramGroup struct {
    Words []string
    Count int
}

func groupAnagramsWithCounts(strs []string) []AnagramGroup {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        sorted := sortString(str)
        groups[sorted] = append(groups[sorted], str)
    }
    
    var result []AnagramGroup
    for _, group := range groups {
        result = append(result, AnagramGroup{
            Words: group,
            Count: len(group),
        })
    }
    
    return result
}
```

#### **Return Sorted Groups**
```go
func groupAnagramsSorted(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        sorted := sortString(str)
        groups[sorted] = append(groups[sorted], str)
    }
    
    var result [][]string
    for _, group := range groups {
        // Sort each group
        sort.Strings(group)
        result = append(result, group)
    }
    
    // Sort result by group size
    sort.Slice(result, func(i, j int) bool {
        return len(result[i]) > len(result[j])
    })
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n×k log k) where n is number of strings and k is average length
- **Space Complexity:** O(n×k)