# Bit Manipulation Pattern

> **Master bit manipulation techniques and bitwise operations with Go implementations**

## 📋 Problems

### **Basic Bit Operations**
- [Single Number](./SingleNumber.md) - Find unique element using XOR
- [Single Number II](./SingleNumberII.md) - Find unique element with triplets
- [Single Number III](./SingleNumberIII.md) - Find two unique elements
- [Missing Number](./MissingNumber.md) - Find missing number using XOR
- [Find the Difference](./FindTheDifference.md) - Find added character

### **Bit Counting and Manipulation**
- [Number of 1 Bits](./NumberOf1Bits.md) - Count set bits (Hamming weight)
- [Counting Bits](./CountingBits.md) - Count bits for all numbers
- [Reverse Bits](./ReverseBits.md) - Reverse bits of a number
- [Power of Two](./PowerOfTwo.md) - Check if number is power of 2
- [Power of Four](./PowerOfFour.md) - Check if number is power of 4

### **Advanced Bit Manipulation**
- [Maximum XOR of Two Numbers](./MaximumXOROfTwoNumbers.md) - Find maximum XOR
- [Subsets](./Subsets.md) - Generate subsets using bit manipulation
- [Gray Code](./GrayCode.md) - Generate Gray code sequence
- [UTF-8 Validation](./UTF8Validation.md) - Validate UTF-8 encoding
- [Bitwise AND of Numbers Range](./BitwiseANDOfNumbersRange.md) - AND operation on range

---

## 🎯 Key Concepts

### **Bitwise Operations in Go**
```go
// Basic bitwise operations
a & b   // AND
a | b   // OR
a ^ b   // XOR
~a      // NOT (complement)
a << n  // Left shift
a >> n  // Right shift
```

### **Common Bit Patterns**
- **XOR Properties**: `a ^ a = 0`, `a ^ 0 = a`, `a ^ b ^ a = b`
- **Power of 2**: `n & (n-1) == 0` for n > 0
- **Set Bit**: `n | (1 << i)`
- **Clear Bit**: `n & ^(1 << i)`
- **Toggle Bit**: `n ^ (1 << i)`
- **Check Bit**: `(n >> i) & 1`

### **Bit Manipulation Techniques**
- **XOR for Finding Unique**: XOR all elements to find unique one
- **Bit Masking**: Use masks to extract specific bits
- **Bit Counting**: Count set bits efficiently
- **Bit Shifting**: Multiply/divide by powers of 2

---

## 🛠️ Go-Specific Tips

### **Bit Operations**
```go
// Count number of 1 bits
func hammingWeight(n uint32) int {
    count := 0
    for n != 0 {
        count++
        n &= n - 1 // Remove the rightmost set bit
    }
    return count
}

// Check if number is power of 2
func isPowerOfTwo(n int) bool {
    return n > 0 && n&(n-1) == 0
}

// Get the rightmost set bit
func getRightmostSetBit(n int) int {
    return n & -n
}
```

### **XOR Properties**
```go
// Find single number in array
func singleNumber(nums []int) int {
    result := 0
    for _, num := range nums {
        result ^= num
    }
    return result
}

// Find missing number
func missingNumber(nums []int) int {
    n := len(nums)
    result := n
    
    for i := 0; i < n; i++ {
        result ^= i ^ nums[i]
    }
    
    return result
}
```

### **Bit Masking**
```go
// Generate all subsets using bit manipulation
func subsets(nums []int) [][]int {
    n := len(nums)
    result := [][]int{}
    
    // Generate all possible subsets
    for i := 0; i < (1 << n); i++ {
        subset := []int{}
        for j := 0; j < n; j++ {
            if (i >> j) & 1 == 1 {
                subset = append(subset, nums[j])
            }
        }
        result = append(result, subset)
    }
    
    return result
}
```

### **Bit Counting**
```go
// Count bits for all numbers from 0 to n
func countBits(n int) []int {
    result := make([]int, n+1)
    
    for i := 1; i <= n; i++ {
        result[i] = result[i>>1] + (i & 1)
    }
    
    return result
}

// Alternative approach using bit manipulation
func countBitsAlternative(n int) []int {
    result := make([]int, n+1)
    
    for i := 0; i <= n; i++ {
        count := 0
        num := i
        for num != 0 {
            count++
            num &= num - 1
        }
        result[i] = count
    }
    
    return result
}
```

---

## 🎯 Interview Tips

### **How to Identify Bit Manipulation Problems**
1. **Unique Elements**: Use XOR to find unique elements
2. **Power of 2**: Check if number is power of 2
3. **Bit Counting**: Count set bits in numbers
4. **Subsets**: Generate all subsets using bit manipulation
5. **Missing Numbers**: Find missing numbers using XOR

### **Common Bit Manipulation Problem Patterns**
- **XOR Problems**: Find unique elements, missing numbers
- **Bit Counting**: Count set bits, generate bit patterns
- **Power Problems**: Check powers of 2, 4, etc.
- **Subset Generation**: Generate all possible subsets
- **Bit Masking**: Extract specific bits from numbers

### **Optimization Tips**
- **Use XOR**: For finding unique elements efficiently
- **Bit Shifting**: For multiplication/division by powers of 2
- **Bit Masking**: For extracting specific bits
- **Brian Kernighan's Algorithm**: For counting set bits
