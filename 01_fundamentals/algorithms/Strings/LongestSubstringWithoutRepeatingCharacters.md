---
# Auto-generated front matter
Title: Longestsubstringwithoutrepeatingcharacters
LastUpdated: 2025-11-06T20:45:58.685716
Tags: []
Status: draft
---

# Longest Substring Without Repeating Characters

### Problem
Given a string `s`, find the length of the longest substring without repeating characters.

**Example:**
```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
```

**Constraints:**
- 0 ≤ s.length ≤ 5 × 10⁴
- s consists of English letters, digits, symbols and spaces

### Explanation

#### **Sliding Window + Hash Map**
- Use two pointers to define sliding window
- Use hash map to track character frequencies
- Expand right pointer and contract left pointer when duplicate found
- Time Complexity: O(n)
- Space Complexity: O(min(m,n)) where m is charset size

### Golang Solution

```go
func lengthOfLongestSubstring(s string) int {
    if len(s) == 0 {
        return 0
    }
    
    charMap := make(map[byte]int)
    left := 0
    maxLen := 0
    
    for right := 0; right < len(s); right++ {
        charMap[s[right]]++
        
        // Contract window while we have duplicates
        for charMap[s[right]] > 1 {
            charMap[s[left]]--
            left++
        }
        
        maxLen = max(maxLen, right-left+1)
    }
    
    return maxLen
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Notes / Variations

#### **Related Problems**
- **Longest Substring with At Most K Distinct Characters**: Find longest substring with K distinct chars
- **Longest Repeating Character Replacement**: Replace characters to get longest substring
- **Minimum Window Substring**: Find minimum window containing all characters
- **Permutation in String**: Check if permutation exists
