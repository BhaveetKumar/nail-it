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
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
```

### Golang Solution

```go
func hammingWeight(num uint32) int {
    count := 0
    
    for num != 0 {
        count += int(num & 1)
        num >>= 1
    }
    
    return count
}
```

### Alternative Solutions

#### **Using Built-in Function**
```go
import "math/bits"

func hammingWeightBuiltin(num uint32) int {
    return bits.OnesCount32(num)
}
```

#### **Using Brian Kernighan's Algorithm**
```go
func hammingWeightKernighan(num uint32) int {
    count := 0
    
    for num != 0 {
        num &= num - 1
        count++
    }
    
    return count
}
```

#### **Using Lookup Table**
```go
func hammingWeightLookup(num uint32) int {
    // Precomputed lookup table for 8-bit numbers
    lookup := []int{
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
    }
    
    return lookup[num&0xFF] + lookup[(num>>8)&0xFF] + lookup[(num>>16)&0xFF] + lookup[(num>>24)&0xFF]
}
```

#### **Using Parallel Count**
```go
func hammingWeightParallel(num uint32) int {
    // Count bits in parallel
    num = num - ((num >> 1) & 0x55555555)
    num = (num & 0x33333333) + ((num >> 2) & 0x33333333)
    num = (num + (num >> 4)) & 0x0F0F0F0F
    num = num + (num >> 8)
    num = num + (num >> 16)
    
    return int(num & 0x3F)
}
```

#### **Return Bit Positions**
```go
func hammingWeightWithPositions(num uint32) (int, []int) {
    count := 0
    var positions []int
    
    for i := 0; i < 32; i++ {
        if num&(1<<i) != 0 {
            count++
            positions = append(positions, i)
        }
    }
    
    return count, positions
}
```

#### **Return Binary Representation**
```go
func hammingWeightWithBinary(num uint32) (int, string) {
    count := 0
    var binary strings.Builder
    
    for i := 31; i >= 0; i-- {
        if num&(1<<i) != 0 {
            count++
            binary.WriteString("1")
        } else {
            binary.WriteString("0")
        }
    }
    
    return count, binary.String()
}
```

#### **Count Bits in Range**
```go
func countBitsInRange(start, end uint32) []int {
    result := make([]int, int(end-start)+1)
    
    for i := start; i <= end; i++ {
        result[i-start] = hammingWeight(i)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(1) for most approaches (32 bits max)
- **Space Complexity:** O(1)