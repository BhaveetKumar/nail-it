# Number of 1 Bits

### Problem
Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

**Example:**
```
Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Input: n = 00000000000000000000000010000000
Output: 1
```

### Golang Solution

```go
func hammingWeight(num uint32) int {
    count := 0
    for num != 0 {
        count++
        num &= num - 1 // Remove the rightmost set bit
    }
    return count
}
```

### Alternative Solutions

#### **Shift Approach**
```go
func hammingWeightShift(num uint32) int {
    count := 0
    for num != 0 {
        if num&1 == 1 {
            count++
        }
        num >>= 1
    }
    return count
}
```

### Complexity
- **Time Complexity:** O(1) - at most 32 iterations
- **Space Complexity:** O(1)
