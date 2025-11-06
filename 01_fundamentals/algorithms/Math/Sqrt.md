---
# Auto-generated front matter
Title: Sqrt
LastUpdated: 2025-11-06T20:45:58.714829
Tags: []
Status: draft
---

# Sqrt(x)

### Problem
Given a non-negative integer `x`, return the square root of `x` rounded down to the nearest integer. The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

- For example, do not use `pow(x, 0.5)` in c++ or `x ** 0.5` in python.

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
        
        if mid == x/mid {
            return mid
        } else if mid < x/mid {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return right
}
```

### Alternative Solutions

#### **Using Newton's Method**
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

#### **Using Linear Search**
```go
func mySqrtLinear(x int) int {
    if x < 2 {
        return x
    }
    
    for i := 1; i <= x/2; i++ {
        if i*i == x {
            return i
        } else if i*i > x {
            return i - 1
        }
    }
    
    return x / 2
}
```

#### **Return with Precision**
```go
type SqrtResult struct {
    IntegerPart int
    DecimalPart float64
    IsPerfect   bool
    Error       float64
}

func mySqrtWithPrecision(x int, precision int) SqrtResult {
    if x < 0 {
        return SqrtResult{Error: -1}
    }
    
    if x < 2 {
        return SqrtResult{
            IntegerPart: x,
            DecimalPart: 0,
            IsPerfect:   true,
        }
    }
    
    // Find integer part
    left, right := 2, x/2
    integerPart := 0
    
    for left <= right {
        mid := left + (right-left)/2
        
        if mid == x/mid {
            integerPart = mid
            break
        } else if mid < x/mid {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    if integerPart == 0 {
        integerPart = right
    }
    
    // Check if perfect square
    isPerfect := integerPart*integerPart == x
    
    // Calculate decimal part
    decimalPart := 0.0
    if !isPerfect {
        decimalPart = calculateDecimalPart(x, integerPart, precision)
    }
    
    return SqrtResult{
        IntegerPart: integerPart,
        DecimalPart: decimalPart,
        IsPerfect:   isPerfect,
    }
}

func calculateDecimalPart(x, integerPart, precision int) float64 {
    result := float64(integerPart)
    increment := 0.1
    
    for i := 0; i < precision; i++ {
        for result*result <= float64(x) {
            result += increment
        }
        result -= increment
        increment /= 10
    }
    
    return result - float64(integerPart)
}
```

#### **Return All Perfect Squares**
```go
func findAllPerfectSquares(x int) []int {
    var perfectSquares []int
    
    for i := 1; i*i <= x; i++ {
        perfectSquares = append(perfectSquares, i*i)
    }
    
    return perfectSquares
}
```

#### **Return with Statistics**
```go
type SqrtStats struct {
    Input        int
    Sqrt         int
    IsPerfect    bool
    PerfectSquares []int
    ClosestPerfect int
    Distance     int
}

func sqrtStatistics(x int) SqrtStats {
    if x < 0 {
        return SqrtStats{Input: x, Sqrt: -1}
    }
    
    sqrt := mySqrt(x)
    isPerfect := sqrt*sqrt == x
    
    perfectSquares := findAllPerfectSquares(x)
    
    closestPerfect := 0
    minDistance := x
    
    for _, perfect := range perfectSquares {
        distance := abs(perfect - x)
        if distance < minDistance {
            minDistance = distance
            closestPerfect = perfect
        }
    }
    
    return SqrtStats{
        Input:           x,
        Sqrt:            sqrt,
        IsPerfect:       isPerfect,
        PerfectSquares:  perfectSquares,
        ClosestPerfect:  closestPerfect,
        Distance:        minDistance,
    }
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```

#### **Return with Range**
```go
type SqrtRange struct {
    Sqrt         int
    LowerBound   int
    UpperBound   int
    IsInRange    bool
    RangeSize    int
}

func sqrtWithRange(x int, lower, upper int) SqrtRange {
    sqrt := mySqrt(x)
    
    return SqrtRange{
        Sqrt:       sqrt,
        LowerBound: lower,
        UpperBound: upper,
        IsInRange:  sqrt >= lower && sqrt <= upper,
        RangeSize:  upper - lower + 1,
    }
}
```

### Complexity
- **Time Complexity:** O(log x) for binary search, O(1) for Newton's method
- **Space Complexity:** O(1)