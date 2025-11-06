---
# Auto-generated front matter
Title: Powerofthree
LastUpdated: 2025-11-06T20:45:58.715739
Tags: []
Status: draft
---

# Power of Three

### Problem
Given an integer `n`, return `true` if it is a power of three. Otherwise, return `false`.

An integer `n` is a power of three, if there exists an integer `x` such that `n == 3^x`.

**Example:**
```
Input: n = 27
Output: true

Input: n = 0
Output: false

Input: n = 9
Output: true

Input: n = 45
Output: false
```

### Golang Solution

```go
func isPowerOfThree(n int) bool {
    if n <= 0 {
        return false
    }
    
    for n % 3 == 0 {
        n /= 3
    }
    
    return n == 1
}
```

### Alternative Solutions

#### **Mathematical Approach**
```go
import "math"

func isPowerOfThreeMath(n int) bool {
    if n <= 0 {
        return false
    }
    
    // 3^19 is the largest power of 3 that fits in 32-bit integer
    return 1162261467 % n == 0
}
```

#### **Logarithm Approach**
```go
import "math"

func isPowerOfThreeLog(n int) bool {
    if n <= 0 {
        return false
    }
    
    log := math.Log(float64(n)) / math.Log(3)
    return math.Abs(log-math.Round(log)) < 1e-10
}
```

#### **Recursive Approach**
```go
func isPowerOfThreeRecursive(n int) bool {
    if n <= 0 {
        return false
    }
    
    if n == 1 {
        return true
    }
    
    if n%3 != 0 {
        return false
    }
    
    return isPowerOfThreeRecursive(n / 3)
}
```

#### **Bit Manipulation**
```go
func isPowerOfThreeBit(n int) bool {
    if n <= 0 {
        return false
    }
    
    // Powers of 3 in binary have specific patterns
    // 3^0 = 1 = 1
    // 3^1 = 3 = 11
    // 3^2 = 9 = 1001
    // 3^3 = 27 = 11011
    // etc.
    
    // Check if n is a power of 3 using bit manipulation
    return n > 0 && 1162261467%n == 0
}
```

#### **String Conversion**
```go
import "strconv"

func isPowerOfThreeString(n int) bool {
    if n <= 0 {
        return false
    }
    
    // Convert to base 3
    base3 := strconv.FormatInt(int64(n), 3)
    
    // Check if it's a 1 followed by all 0s
    if base3[0] != '1' {
        return false
    }
    
    for i := 1; i < len(base3); i++ {
        if base3[i] != '0' {
            return false
        }
    }
    
    return true
}
```

#### **Iterative with Multiplication**
```go
func isPowerOfThreeIterative(n int) bool {
    if n <= 0 {
        return false
    }
    
    power := 1
    for power < n {
        power *= 3
    }
    
    return power == n
}
```

### Complexity
- **Time Complexity:** O(log n) for division, O(1) for mathematical approaches
- **Space Complexity:** O(1)