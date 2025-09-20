# Bit Manipulation Pattern

> **Master bit manipulation techniques and bitwise operations with Go implementations**

## 📋 Problems

### **Basic Bit Operations**

- [Single Number](SingleNumber.md) - Find unique element using XOR
- [Single Number II](SingleNumberII.md) - Find unique element with triplets
- [Single Number III](SingleNumberIII.md) - Find two unique elements
- [Missing Number](MissingNumber.md) - Find missing number using XOR
- [Find the Difference](FindTheDifference.md) - Find added character

### **Bit Counting and Manipulation**

- [Number of 1 Bits](NumberOf1Bits.md) - Count set bits (Hamming weight)
- [Counting Bits](CountingBits.md) - Count bits for all numbers
- [Reverse Bits](ReverseBits.md) - Reverse bits of a number
- [Power of Two](PowerOfTwo.md) - Check if number is power of 2
- [Power of Four](PowerOfFour.md) - Check if number is power of 4

### **Advanced Bit Manipulation**

- [Maximum XOR of Two Numbers](MaximumXOROfTwoNumbers.md) - Find maximum XOR
- [Subsets](Subsets.md) - Generate subsets using bit manipulation
- [Gray Code](GrayCode.md) - Generate Gray code sequence
- [UTF-8 Validation](UTF8Validation.md) - Validate UTF-8 encoding
- [Bitwise AND of Numbers Range](BitwiseANDOfNumbersRange.md) - AND operation on range

---

## 🎯 Key Concepts

### **Bitwise Operations in Go**

**Detailed Explanation:**
Bit manipulation is a powerful technique that operates directly on the binary representation of numbers. In Go, bitwise operations are fundamental tools for solving problems that require working with individual bits or performing operations at the bit level.

**Basic Bitwise Operations:**

```go
// Basic bitwise operations
a & b   // AND - 1 if both bits are 1, 0 otherwise
a | b   // OR - 1 if either bit is 1, 0 otherwise
a ^ b   // XOR - 1 if bits are different, 0 if same
~a      // NOT (complement) - flip all bits
a << n  // Left shift - multiply by 2^n
a >> n  // Right shift - divide by 2^n
```

**Operation Details:**

- **AND (&)**: Used for masking, checking if specific bits are set
- **OR (|)**: Used for setting specific bits
- **XOR (^)**: Used for toggling bits, finding differences
- **NOT (~)**: Used for flipping all bits (be careful with signed integers)
- **Left Shift (<<)**: Used for multiplication by powers of 2
- **Right Shift (>>)**: Used for division by powers of 2

**Go-Specific Considerations:**

- **Unsigned Integers**: Use `uint32`, `uint64` for bit manipulation to avoid sign extension
- **Bit Size**: Go integers are 32 or 64 bits depending on architecture
- **Overflow**: Be careful with shifts that might cause overflow
- **Performance**: Bit operations are extremely fast, often faster than arithmetic operations

### **Common Bit Patterns**

**Detailed Explanation:**
Understanding common bit patterns is crucial for solving bit manipulation problems efficiently. These patterns form the foundation for more complex bit manipulation algorithms.

**XOR Properties:**

- **Identity**: `a ^ 0 = a` - XOR with 0 returns the original number
- **Self-Inverse**: `a ^ a = 0` - XOR with itself returns 0
- **Commutative**: `a ^ b = b ^ a` - Order doesn't matter
- **Associative**: `(a ^ b) ^ c = a ^ (b ^ c)` - Grouping doesn't matter
- **Cancellation**: `a ^ b ^ a = b` - XOR cancels out duplicate elements

**Power of 2 Detection:**

- **Pattern**: `n & (n-1) == 0` for n > 0
- **Why it works**: Powers of 2 have exactly one bit set
- **Example**: 8 (1000) & 7 (0111) = 0
- **Use case**: Check if number is power of 2, find highest power of 2

**Bit Manipulation Operations:**

- **Set Bit**: `n | (1 << i)` - Set bit at position i
- **Clear Bit**: `n & ^(1 << i)` - Clear bit at position i
- **Toggle Bit**: `n ^ (1 << i)` - Toggle bit at position i
- **Check Bit**: `(n >> i) & 1` - Check if bit at position i is set
- **Get Rightmost Set Bit**: `n & -n` - Get the rightmost set bit

**Advanced Patterns:**

- **Isolate Rightmost Set Bit**: `n & -n`
- **Remove Rightmost Set Bit**: `n & (n-1)`
- **Get All Set Bits**: Use `n & (n-1)` in a loop
- **Check if Even/Odd**: `n & 1` - 0 for even, 1 for odd

### **Bit Manipulation Techniques**

**Detailed Explanation:**
Bit manipulation techniques are systematic approaches to solving problems using bitwise operations. These techniques can often provide O(1) or O(log n) solutions to problems that would otherwise require O(n) time.

**XOR for Finding Unique Elements:**

- **Principle**: XOR has the property that `a ^ a = 0` and `a ^ 0 = a`
- **Application**: Find single unique element in array of duplicates
- **Algorithm**: XOR all elements together
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Example**: `[2,1,2,1,3]` → `2^1^2^1^3 = 3`

**Bit Masking:**

- **Purpose**: Extract or manipulate specific bits
- **Techniques**: Use bit masks to isolate specific bit positions
- **Applications**: Subset generation, bit extraction, bit manipulation
- **Example**: Extract bits 2-4 using mask `0b11100`

**Bit Counting (Hamming Weight):**

- **Purpose**: Count number of set bits in a number
- **Brian Kernighan's Algorithm**: `n &= n-1` removes rightmost set bit
- **Time Complexity**: O(number of set bits)
- **Space Complexity**: O(1)
- **Applications**: Hamming distance, bit manipulation problems

**Bit Shifting:**

- **Left Shift**: Multiply by powers of 2
- **Right Shift**: Divide by powers of 2
- **Applications**: Fast multiplication/division, bit manipulation
- **Caution**: Watch for overflow and sign extension

**Discussion Questions & Answers:**

**Q1: How do you optimize bit manipulation operations for performance in Go?**

**Answer:** Performance optimization strategies:

- **Use Unsigned Types**: Use `uint32`, `uint64` to avoid sign extension issues
- **Avoid Unnecessary Conversions**: Work with bits directly when possible
- **Use Bit Masks**: Pre-compute masks for repeated operations
- **Loop Unrolling**: Unroll small loops for bit counting operations
- **Compiler Optimizations**: Let the compiler optimize bit operations
- **Memory Access**: Consider cache locality for bit array operations
- **Branch Prediction**: Use bit operations to avoid branches when possible
- **Profiling**: Use `go tool pprof` to identify bit manipulation bottlenecks

**Q2: What are the common pitfalls when working with bit manipulation in Go?**

**Answer:** Common pitfalls include:

- **Sign Extension**: Right shift on signed integers extends the sign bit
- **Integer Overflow**: Left shift can cause overflow for large numbers
- **Bit Order**: Confusion about bit numbering (0-indexed from right)
- **Type Conversion**: Implicit conversions between signed and unsigned types
- **Bit Size**: Assuming specific bit sizes across different architectures
- **Operator Precedence**: Bitwise operators have different precedence than arithmetic
- **Negative Numbers**: Two's complement representation for negative numbers
- **Floating Point**: Bit operations don't work directly on floating point numbers

**Q3: How do you implement efficient bit manipulation algorithms for large datasets?**

**Answer:** Large dataset optimization:

- **Bit Arrays**: Use bit arrays for memory-efficient storage
- **Parallel Processing**: Use goroutines for parallel bit operations
- **SIMD Instructions**: Leverage CPU SIMD instructions when available
- **Memory Pooling**: Reuse bit arrays to reduce garbage collection
- **Chunking**: Process data in chunks to improve cache locality
- **Compression**: Use bit compression techniques for sparse data
- **Lazy Evaluation**: Compute bit operations only when needed
- **Caching**: Cache frequently computed bit patterns

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
