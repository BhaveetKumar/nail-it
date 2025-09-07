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
func isPowerOfThreeMath(n int) bool {
    if n <= 0 {
        return false
    }
    
    // 3^19 is the largest power of 3 that fits in 32-bit integer
    return 1162261467 % n == 0
}
```

#### **Logarithmic Approach**
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

### Complexity
- **Time Complexity:** O(log n) for loop, O(1) for mathematical
- **Space Complexity:** O(1)
