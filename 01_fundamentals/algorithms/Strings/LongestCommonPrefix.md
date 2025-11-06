---
# Auto-generated front matter
Title: Longestcommonprefix
LastUpdated: 2025-11-06T20:45:58.688478
Tags: []
Status: draft
---

# Longest Common Prefix

### Problem
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string `""`.

**Example:**
```
Input: strs = ["flower","flow","flight"]
Output: "fl"

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
```

### Golang Solution

```go
func longestCommonPrefix(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    prefix := strs[0]
    
    for i := 1; i < len(strs); i++ {
        for len(prefix) > 0 && !strings.HasPrefix(strs[i], prefix) {
            prefix = prefix[:len(prefix)-1]
        }
        
        if prefix == "" {
            break
        }
    }
    
    return prefix
}
```

### Alternative Solutions

#### **Vertical Scanning**
```go
func longestCommonPrefixVertical(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    for i := 0; i < len(strs[0]); i++ {
        char := strs[0][i]
        
        for j := 1; j < len(strs); j++ {
            if i >= len(strs[j]) || strs[j][i] != char {
                return strs[0][:i]
            }
        }
    }
    
    return strs[0]
}
```

#### **Binary Search**
```go
func longestCommonPrefixBinarySearch(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    minLen := len(strs[0])
    for _, str := range strs {
        if len(str) < minLen {
            minLen = len(str)
        }
    }
    
    left, right := 0, minLen
    
    for left < right {
        mid := left + (right-left+1)/2
        
        if isCommonPrefix(strs, mid) {
            left = mid
        } else {
            right = mid - 1
        }
    }
    
    return strs[0][:left]
}

func isCommonPrefix(strs []string, length int) bool {
    prefix := strs[0][:length]
    
    for i := 1; i < len(strs); i++ {
        if !strings.HasPrefix(strs[i], prefix) {
            return false
        }
    }
    
    return true
}
```

#### **Divide and Conquer**
```go
func longestCommonPrefixDivideConquer(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    return divideConquer(strs, 0, len(strs)-1)
}

func divideConquer(strs []string, left, right int) string {
    if left == right {
        return strs[left]
    }
    
    mid := left + (right-left)/2
    leftPrefix := divideConquer(strs, left, mid)
    rightPrefix := divideConquer(strs, mid+1, right)
    
    return commonPrefix(leftPrefix, rightPrefix)
}

func commonPrefix(str1, str2 string) string {
    minLen := len(str1)
    if len(str2) < minLen {
        minLen = len(str2)
    }
    
    for i := 0; i < minLen; i++ {
        if str1[i] != str2[i] {
            return str1[:i]
        }
    }
    
    return str1[:minLen]
}
```

#### **Using Trie**
```go
type TrieNode struct {
    children map[byte]*TrieNode
    isEnd    bool
}

func longestCommonPrefixTrie(strs []string) string {
    if len(strs) == 0 {
        return ""
    }
    
    root := &TrieNode{children: make(map[byte]*TrieNode)}
    
    // Insert all strings into trie
    for _, str := range strs {
        insertTrie(root, str)
    }
    
    // Find common prefix
    return findCommonPrefix(root, len(strs))
}

func insertTrie(root *TrieNode, str string) {
    node := root
    for i := 0; i < len(str); i++ {
        char := str[i]
        if node.children[char] == nil {
            node.children[char] = &TrieNode{children: make(map[byte]*TrieNode)}
        }
        node = node.children[char]
    }
    node.isEnd = true
}

func findCommonPrefix(root *TrieNode, totalStrings int) string {
    var result strings.Builder
    node := root
    
    for len(node.children) == 1 && !node.isEnd {
        for char, child := range node.children {
            result.WriteByte(char)
            node = child
            break
        }
    }
    
    return result.String()
}
```

### Complexity
- **Time Complexity:** O(S) where S is sum of all characters, O(n×m) for vertical scanning
- **Space Complexity:** O(1) for most approaches, O(m×log n) for divide and conquer