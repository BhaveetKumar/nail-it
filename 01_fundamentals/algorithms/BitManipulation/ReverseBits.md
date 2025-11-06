---
# Auto-generated front matter
Title: Reversebits
LastUpdated: 2025-11-06T20:45:58.689836
Tags: []
Status: draft
---

# Reverse Bits

### Problem
Reverse bits of a given 32 bits unsigned integer.

**Example:**
```
Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.

Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
```

### Golang Solution

```go
func reverseBits(num uint32) uint32 {
    result := uint32(0)
    
    for i := 0; i < 32; i++ {
        result = (result << 1) | (num & 1)
        num >>= 1
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Bit Manipulation**
```go
func reverseBitsBitManipulation(num uint32) uint32 {
    result := uint32(0)
    
    for i := 0; i < 32; i++ {
        if num&(1<<i) != 0 {
            result |= 1 << (31 - i)
        }
    }
    
    return result
}
```

#### **Using Lookup Table**
```go
func reverseBitsLookup(num uint32) uint32 {
    // Precomputed lookup table for 8-bit numbers
    lookup := []uint32{
        0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
        8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
        4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
        12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
        2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
        10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
        6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
        14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
        1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
        9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
        5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
        13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
        3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
        11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
        7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
        15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255,
    }
    
    return (lookup[num&0xFF] << 24) | (lookup[(num>>8)&0xFF] << 16) | (lookup[(num>>16)&0xFF] << 8) | lookup[(num>>24)&0xFF]
}
```

#### **Using String Conversion**
```go
import "strconv"

func reverseBitsString(num uint32) uint32 {
    binary := fmt.Sprintf("%032b", num)
    reversed := ""
    
    for i := len(binary) - 1; i >= 0; i-- {
        reversed += string(binary[i])
    }
    
    result, _ := strconv.ParseUint(reversed, 2, 32)
    return uint32(result)
}
```

#### **Return with Binary Representation**
```go
type ReverseResult struct {
    Original     uint32
    Reversed     uint32
    OriginalBin  string
    ReversedBin  string
    BitCount     int
}

func reverseBitsWithBinary(num uint32) ReverseResult {
    reversed := reverseBits(num)
    
    return ReverseResult{
        Original:    num,
        Reversed:    reversed,
        OriginalBin: fmt.Sprintf("%032b", num),
        ReversedBin: fmt.Sprintf("%032b", reversed),
        BitCount:    32,
    }
}
```

#### **Return All Bit Operations**
```go
type BitOperations struct {
    Original     uint32
    Reversed     uint32
    Complement   uint32
    LeftShift    uint32
    RightShift   uint32
    RotateLeft   uint32
    RotateRight  uint32
}

func allBitOperations(num uint32) BitOperations {
    reversed := reverseBits(num)
    complement := ^num
    leftShift := num << 1
    rightShift := num >> 1
    rotateLeft := (num << 1) | (num >> 31)
    rotateRight := (num >> 1) | (num << 31)
    
    return BitOperations{
        Original:    num,
        Reversed:    reversed,
        Complement:  complement,
        LeftShift:   leftShift,
        RightShift:  rightShift,
        RotateLeft:  rotateLeft,
        RotateRight: rotateRight,
    }
}
```

#### **Return Bit Statistics**
```go
type BitStats struct {
    Original     uint32
    Reversed     uint32
    BitCount     int
    SetBits      int
    ClearBits    int
    LeadingZeros int
    TrailingZeros int
    Parity       int
}

func bitStatistics(num uint32) BitStats {
    reversed := reverseBits(num)
    
    setBits := 0
    for i := 0; i < 32; i++ {
        if num&(1<<i) != 0 {
            setBits++
        }
    }
    
    leadingZeros := 0
    for i := 31; i >= 0; i-- {
        if num&(1<<i) == 0 {
            leadingZeros++
        } else {
            break
        }
    }
    
    trailingZeros := 0
    for i := 0; i < 32; i++ {
        if num&(1<<i) == 0 {
            trailingZeros++
        } else {
            break
        }
    }
    
    parity := 0
    for i := 0; i < 32; i++ {
        if num&(1<<i) != 0 {
            parity ^= 1
        }
    }
    
    return BitStats{
        Original:      num,
        Reversed:      reversed,
        BitCount:      32,
        SetBits:       setBits,
        ClearBits:     32 - setBits,
        LeadingZeros:  leadingZeros,
        TrailingZeros: trailingZeros,
        Parity:        parity,
    }
}
```

### Complexity
- **Time Complexity:** O(1) for all approaches (32 bits max)
- **Space Complexity:** O(1)