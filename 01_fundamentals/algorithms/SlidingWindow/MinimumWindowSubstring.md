# Minimum Window Substring

### Problem
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the minimum window substring of `s` such that every character in `t` (including duplicates) is included in the window. If there is no such window, return the empty string `""`.

**Example:**
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
```

### Golang Solution

```go
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }
    
    targetCount := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        targetCount[t[i]]++
    }
    
    left := 0
    minLen := len(s) + 1
    minStart := 0
    matched := 0
    
    for right := 0; right < len(s); right++ {
        if targetCount[s[right]] > 0 {
            matched++
        }
        targetCount[s[right]]--
        
        for matched == len(t) {
            if right-left+1 < minLen {
                minLen = right - left + 1
                minStart = left
            }
            
            targetCount[s[left]]++
            if targetCount[s[left]] > 0 {
                matched--
            }
            left++
        }
    }
    
    if minLen > len(s) {
        return ""
    }
    
    return s[minStart : minStart+minLen]
}
```

### Complexity
- **Time Complexity:** O(|s| + |t|)
- **Space Complexity:** O(|s| + |t|)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
