# Sqrt(x)

### Problem
Given a non-negative integer `x`, return the square root of `x` rounded down to the nearest integer. The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

**Example:**
```
Input: x = 4
Output: 2

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
```

### Golang Solution

```go
func mySqrt(x int) int {
    if x < 2 {
        return x
    }
    
    left, right := 2, x/2
    
    for left <= right {
        mid := left + (right-left)/2
        square := mid * mid
        
        if square == x {
            return mid
        } else if square < x {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return right
}
```

### Alternative Solutions

#### **Newton's Method**
```go
func mySqrtNewton(x int) int {
    if x < 2 {
        return x
    }
    
    r := x
    for r*r > x {
        r = (r + x/r) / 2
    }
    
    return r
}
```

### Complexity
- **Time Complexity:** O(log x) for binary search, O(log log x) for Newton's method
- **Space Complexity:** O(1)
