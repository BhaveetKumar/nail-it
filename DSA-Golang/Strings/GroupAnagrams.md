# Group Anagrams

### Problem
Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.

**Example:**
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### Golang Solution

```go
func groupAnagrams(strs []string) [][]string {
    groups := make(map[string][]string)
    
    for _, str := range strs {
        key := getKey(str)
        groups[key] = append(groups[key], str)
    }
    
    result := make([][]string, 0, len(groups))
    for _, group := range groups {
        result = append(result, group)
    }
    
    return result
}

func getKey(s string) string {
    chars := []rune(s)
    sort.Slice(chars, func(i, j int) bool {
        return chars[i] < chars[j]
    })
    return string(chars)
}
```

### Complexity
- **Time Complexity:** O(n × m log m) where n is number of strings, m is average length
- **Space Complexity:** O(n × m)
