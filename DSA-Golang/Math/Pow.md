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
        if n%2 == 1 {
            result *= x
        }
        x *= x
        n /= 2
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
    }
    
    return x * myPowRecursive(x, n-1)
}
```

#### **Using Math Package**
```go
import "math"

func myPowBuiltin(x float64, n int) float64 {
    return math.Pow(x, float64(n))
}
```

#### **Iterative with Bit Manipulation**
```go
func myPowBitManipulation(x float64, n int) float64 {
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

#### **Return with Steps**
```go
type PowResult struct {
    Result float64
    Steps  int
}

func myPowWithSteps(x float64, n int) PowResult {
    if n == 0 {
        return PowResult{Result: 1, Steps: 0}
    }
    
    if n < 0 {
        x = 1 / x
        n = -n
    }
    
    result := 1.0
    steps := 0
    
    for n > 0 {
        steps++
        if n%2 == 1 {
            result *= x
        }
        x *= x
        n /= 2
    }
    
    return PowResult{Result: result, Steps: steps}
}
```

#### **Return All Powers in Range**
```go
func allPowers(x float64, start, end int) []float64 {
    var result []float64
    
    for i := start; i <= end; i++ {
        result = append(result, myPow(x, i))
    }
    
    return result
}
```

#### **Return Power Series**
```go
func powerSeries(x float64, terms int) []float64 {
    var result []float64
    
    for i := 0; i < terms; i++ {
        result = append(result, myPow(x, i))
    }
    
    return result
}
```

#### **Return with Validation**
```go
type PowValidation struct {
    Result    float64
    IsValid   bool
    Error     string
    Precision float64
}

func myPowWithValidation(x float64, n int) PowValidation {
    if x == 0 && n < 0 {
        return PowValidation{
            Result:  0,
            IsValid: false,
            Error:   "Division by zero",
        }
    }
    
    if n == 0 {
        return PowValidation{
            Result:    1,
            IsValid:   true,
            Precision: 0,
        }
    }
    
    result := myPow(x, n)
    
    // Check for overflow
    if math.IsInf(result, 0) {
        return PowValidation{
            Result:  result,
            IsValid: false,
            Error:   "Overflow",
        }
    }
    
    return PowValidation{
        Result:    result,
        IsValid:   true,
        Precision: 1e-10,
    }
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1) for iterative, O(log n) for recursive