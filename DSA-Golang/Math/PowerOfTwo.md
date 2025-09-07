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
    if n <= 0 {
        return false
    }
    
    return n&(n-1) == 0
}
```

### Alternative Solutions

#### **Using Division**
```go
func isPowerOfTwoDivision(n int) bool {
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

#### **Using Logarithm**
```go
import "math"

func isPowerOfTwoLog(n int) bool {
    if n <= 0 {
        return false
    }
    
    log := math.Log2(float64(n))
    return log == math.Floor(log)
}
```

#### **Using Recursion**
```go
func isPowerOfTwoRecursive(n int) bool {
    if n <= 0 {
        return false
    }
    
    if n == 1 {
        return true
    }
    
    if n%2 != 0 {
        return false
    }
    
    return isPowerOfTwoRecursive(n / 2)
}
```

#### **Using Bit Count**
```go
func isPowerOfTwoBitCount(n int) bool {
    if n <= 0 {
        return false
    }
    
    count := 0
    for n > 0 {
        count += n & 1
        n >>= 1
    }
    
    return count == 1
}
```

#### **Using Set**
```go
func isPowerOfTwoSet(n int) bool {
    if n <= 0 {
        return false
    }
    
    powers := map[int]bool{
        1: true, 2: true, 4: true, 8: true, 16: true, 32: true, 64: true, 128: true,
        256: true, 512: true, 1024: true, 2048: true, 4096: true, 8192: true,
        16384: true, 32768: true, 65536: true, 131072: true, 262144: true,
        524288: true, 1048576: true, 2097152: true, 4194304: true, 8388608: true,
        16777216: true, 33554432: true, 67108864: true, 134217728: true,
        268435456: true, 536870912: true, 1073741824: true,
    }
    
    return powers[n]
}
```

#### **Return Power Value**
```go
func isPowerOfTwoWithValue(n int) (bool, int) {
    if n <= 0 {
        return false, -1
    }
    
    if n&(n-1) != 0 {
        return false, -1
    }
    
    power := 0
    temp := n
    for temp > 1 {
        temp >>= 1
        power++
    }
    
    return true, power
}
```

### Complexity
- **Time Complexity:** O(1) for bit manipulation, O(log n) for division
- **Space Complexity:** O(1)
