# Pow(x, n)

### Problem
Implement `pow(x, n)`, which calculates `x` raised to the power `n` (i.e., x^n).

**Example:**
```
Input: x = 2.00000, n = 10
Output: 1024.00000

Input: x = 2.10000, n = 3
Output: 9.26100

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2^-2 = 1/2^2 = 1/4 = 0.25
```

### Golang Solution

```go
func myPow(x float64, n int) float64 {
    if n == 0 {
        return 1
    }
    
    if n < 0 {
        x = 1 / x
        n = -n
    }
    
    result := 1.0
    for n > 0 {
        if n&1 == 1 {
            result *= x
        }
        x *= x
        n >>= 1
    }
    
    return result
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func myPowRecursive(x float64, n int) float64 {
    if n == 0 {
        return 1
    }
    
    if n < 0 {
        return 1 / myPowRecursive(x, -n)
    }
    
    if n%2 == 0 {
        half := myPowRecursive(x, n/2)
        return half * half
    } else {
        return x * myPowRecursive(x, n-1)
    }
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1) for iterative, O(log n) for recursive
