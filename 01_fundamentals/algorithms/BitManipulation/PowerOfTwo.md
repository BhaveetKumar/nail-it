# Power of Two

### Problem
Given an integer `n`, return `true` if it is a power of two. Otherwise, return `false`.

An integer `n` is a power of two, if there exists an integer `x` such that `n == 2^x`.

**Example:**
```
Input: n = 1
Output: true
Explanation: 2^0 = 1

Input: n = 16
Output: true
Explanation: 2^4 = 16

Input: n = 3
Output: false
```

### Golang Solution

```go
func isPowerOfTwo(n int) bool {
    return n > 0 && n&(n-1) == 0
}
```

### Alternative Solutions

#### **Loop Approach**
```go
func isPowerOfTwoLoop(n int) bool {
    if n <= 0 {
        return false
    }
    
    for n > 1 {
        if n%2 != 0 {
            return false
        }
        n /= 2
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(1) for bit manipulation, O(log n) for loop
- **Space Complexity:** O(1)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
