# Sum of Two Integers

### Problem
Given two integers `a` and `b`, return the sum of the two integers without using the operators `+` and `-`.

**Example:**
```
Input: a = 1, b = 2
Output: 3

Input: a = 2, b = 3
Output: 5
```

### Golang Solution

```go
func getSum(a int, b int) int {
    for b != 0 {
        carry := a & b
        a = a ^ b
        b = carry << 1
    }
    return a
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func getSumRecursive(a int, b int) int {
    if b == 0 {
        return a
    }
    return getSumRecursive(a^b, (a&b)<<1)
}
```

#### **Handle Negative Numbers**
```go
func getSumWithNegatives(a int, b int) int {
    // Convert to unsigned to handle negative numbers
    ua := uint32(a)
    ub := uint32(b)
    
    for ub != 0 {
        carry := ua & ub
        ua = ua ^ ub
        ub = carry << 1
    }
    
    return int(int32(ua))
}
```

### Complexity
- **Time Complexity:** O(1) - constant number of iterations
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
