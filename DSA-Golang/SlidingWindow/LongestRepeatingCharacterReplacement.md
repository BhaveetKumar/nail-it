# Longest Repeating Character Replacement

### Problem
You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English letter. You can perform this operation at most `k` times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

**Example:**
```
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
```

### Golang Solution

```go
func characterReplacement(s string, k int) int {
    count := make(map[byte]int)
    left := 0
    maxCount := 0
    maxLength := 0
    
    for right := 0; right < len(s); right++ {
        count[s[right]]++
        maxCount = max(maxCount, count[s[right]])
        
        if right-left+1-maxCount > k {
            count[s[left]]--
            left++
        }
        
        maxLength = max(maxLength, right-left+1)
    }
    
    return maxLength
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
