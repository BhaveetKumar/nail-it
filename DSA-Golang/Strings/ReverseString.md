# Reverse String

### Problem
Write a function that reverses a string. The input string is given as an array of characters `s`.

You must do this by modifying the input array in-place with `O(1)` extra memory.

**Example:**
```
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
```

### Golang Solution

```go
func reverseString(s []byte) {
    left, right := 0, len(s)-1
    
    for left < right {
        s[left], s[right] = s[right], s[left]
        left++
        right--
    }
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func reverseStringRecursive(s []byte) {
    reverseHelper(s, 0, len(s)-1)
}

func reverseHelper(s []byte, left, right int) {
    if left >= right {
        return
    }
    
    s[left], s[right] = s[right], s[left]
    reverseHelper(s, left+1, right-1)
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive
