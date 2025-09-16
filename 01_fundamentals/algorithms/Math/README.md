# Math Pattern

> **Master mathematical algorithms and number theory with Go implementations**

## 📋 Problems

### **Number Theory**

- [Pow(x, n)](Pow.md/) - Calculate x raised to power n
- [Sqrt(x)](Sqrt.md/) - Calculate square root
- [Valid Perfect Square](ValidPerfectSquare.md/) - Check if number is perfect square
- [Power of Three](PowerOfThree.md/) - Check if number is power of 3
- [Roman to Integer](RomanToInteger.md/) - Convert Roman numerals to integer

### **Combinatorics**

- [Pascal's Triangle](PascalsTriangle.md/) - Generate Pascal's triangle
- [Pascal's Triangle II](PascalsTriangleII.md/) - Get specific row
- [Unique Paths](UniquePaths.md/) - Count paths in grid
- [Unique Paths II](UniquePathsII.md/) - Count paths with obstacles
- [Climbing Stairs](ClimbingStairs.md/) - Count ways to climb stairs

### **Arithmetic**

- [Add Binary](AddBinary.md/) - Add two binary strings
- [Multiply Strings](MultiplyStrings.md/) - Multiply two strings
- [Plus One](PlusOne.md/) - Add one to number represented as array
- [Factorial Trailing Zeroes](FactorialTrailingZeroes.md/) - Count trailing zeros in factorial
- [Excel Sheet Column Title](ExcelSheetColumnTitle.md/) - Convert number to Excel column

---

## 🎯 Key Concepts

### **Mathematical Operations**

**Detailed Explanation:**
Mathematical operations in programming often require efficient algorithms to handle large numbers, avoid overflow, and optimize performance. Understanding the mathematical principles behind these operations is crucial for solving complex problems.

**1. Exponentiation:**

- **Fast Power Calculation**: Use binary exponentiation (exponentiation by squaring) for O(log n) time complexity
- **Algorithm**: Break down the exponent into binary representation and compute powers of 2
- **Example**: x^13 = x^8 _ x^4 _ x^1 (13 = 8 + 4 + 1 in binary)
- **Benefits**: Much faster than naive O(n) approach, especially for large exponents
- **Applications**: Cryptography, modular arithmetic, matrix exponentiation

**2. Square Root:**

- **Binary Search Approach**: Use binary search to find square root with O(log n) time complexity
- **Algorithm**: Search in range [0, x] for the largest number whose square is ≤ x
- **Precision**: Can be extended to find square root with decimal precision
- **Applications**: Distance calculations, geometric problems, numerical analysis
- **Alternative**: Newton's method for faster convergence

**3. Modular Arithmetic:**

- **Purpose**: Handle large numbers by working with remainders
- **Properties**: (a + b) mod m = ((a mod m) + (b mod m)) mod m
- **Applications**: Cryptography, hash functions, avoiding overflow
- **Efficient Operations**: Use modular exponentiation for large powers
- **Prime Modulus**: Special properties when working with prime numbers

**4. Combinatorics:**

- **Permutations**: Arrangements of objects where order matters
- **Combinations**: Selections of objects where order doesn't matter
- **Formulas**: n! for permutations, C(n,k) = n!/(k!(n-k)!) for combinations
- **Applications**: Counting problems, probability, optimization
- **Dynamic Programming**: Use DP for large combinatorial calculations

### **Common Patterns**

**Detailed Explanation:**
Mathematical problems often follow specific patterns that can be recognized and solved using established algorithms and techniques.

**1. Binary Search:**

- **When to Use**: For problems involving finding a value in a sorted range
- **Applications**: Square root, power calculations, finding optimal values
- **Time Complexity**: O(log n)
- **Implementation**: Use left and right pointers, adjust based on comparison
- **Edge Cases**: Handle boundary conditions and precision requirements

**2. Dynamic Programming:**

- **When to Use**: For counting problems with overlapping subproblems
- **Applications**: Unique paths, climbing stairs, combinatorial problems
- **Time Complexity**: O(n) or O(n²) depending on problem
- **Space Optimization**: Can often optimize space from O(n) to O(1)
- **Pattern Recognition**: Look for optimal substructure and overlapping subproblems

**3. String Manipulation:**

- **When to Use**: For arithmetic operations on large numbers represented as strings
- **Applications**: Add binary, multiply strings, large number arithmetic
- **Implementation**: Simulate manual arithmetic operations
- **Edge Cases**: Handle carry operations, leading zeros, negative numbers
- **Optimization**: Use efficient string operations and avoid unnecessary conversions

**4. Number Theory:**

- **Prime Numbers**: Efficient algorithms for prime checking and generation
- **GCD (Greatest Common Divisor)**: Euclidean algorithm for finding GCD
- **LCM (Least Common Multiple)**: Calculate using GCD formula
- **Applications**: Cryptography, optimization, mathematical proofs
- **Advanced Topics**: Modular arithmetic, Chinese remainder theorem

**Advanced Mathematical Concepts:**

- **Matrix Operations**: Matrix multiplication, exponentiation, transformations
- **Geometric Algorithms**: Distance calculations, area computations, convex hull
- **Probability**: Combinatorial probability, conditional probability
- **Statistics**: Mean, median, mode, standard deviation
- **Numerical Methods**: Root finding, integration, differential equations

**Discussion Questions & Answers:**

**Q1: How do you optimize mathematical algorithms for performance in Go?**

**Answer:** Performance optimization strategies:

- **Algorithm Selection**: Choose the most efficient algorithm for the problem (e.g., binary exponentiation vs naive)
- **Data Types**: Use appropriate data types (int64 for large numbers, float64 for precision)
- **Avoid Overflow**: Use modular arithmetic or larger data types to prevent overflow
- **Memory Management**: Reuse variables and avoid unnecessary allocations
- **Bit Manipulation**: Use bit operations for power-of-2 calculations
- **Caching**: Cache frequently computed values (factorials, powers)
- **Parallel Processing**: Use goroutines for independent calculations when applicable
- **Compiler Optimizations**: Let the Go compiler optimize mathematical operations
- **Profiling**: Use `go tool pprof` to identify performance bottlenecks
- **Mathematical Insights**: Use mathematical properties to simplify calculations

**Q2: What are the common pitfalls when implementing mathematical algorithms in Go?**

**Answer:** Common pitfalls include:

- **Integer Overflow**: Not handling large numbers that exceed int64 limits
- **Floating Point Precision**: Issues with floating point arithmetic and precision
- **Division by Zero**: Not checking for zero divisors in division operations
- **Negative Numbers**: Incorrect handling of negative numbers in power calculations
- **Edge Cases**: Not handling edge cases like 0, 1, negative numbers
- **Type Conversions**: Issues with type conversions between int and float
- **Modulo Operations**: Incorrect handling of negative numbers in modulo operations
- **String Operations**: Inefficient string operations for large number arithmetic
- **Memory Leaks**: Not properly managing memory for large calculations
- **Algorithm Complexity**: Choosing inefficient algorithms for large inputs

**Q3: How do you handle large numbers and precision in mathematical calculations?**

**Answer:** Large number and precision handling:

- **Big Integer**: Use `math/big` package for arbitrary precision integers
- **Big Float**: Use `math/big` package for arbitrary precision floating point
- **Modular Arithmetic**: Use modular arithmetic to keep numbers manageable
- **String Representation**: Use string representation for very large numbers
- **Scientific Notation**: Use scientific notation for very large or very small numbers
- **Precision Control**: Set appropriate precision for floating point calculations
- **Rounding**: Implement proper rounding strategies for decimal calculations
- **Error Handling**: Implement proper error handling for overflow and underflow
- **Validation**: Validate input ranges and handle edge cases
- **Testing**: Test with edge cases and large numbers to ensure correctness

---

## 🛠️ Go-Specific Tips

### **Fast Power Calculation**

```go
func pow(x float64, n int) float64 {
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

### **Square Root with Binary Search**

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

### **GCD and LCM**

```go
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func lcm(a, b int) int {
    return a * b / gcd(a, b)
}
```

---

## 🎯 Interview Tips

### **How to Identify Math Problems**

1. **Number Operations**: Exponentiation, square root, factorial
2. **Counting Problems**: Paths, combinations, permutations
3. **String Arithmetic**: Large number operations
4. **Number Theory**: Prime numbers, divisibility

### **Common Math Problem Patterns**

- **Fast Power**: Use binary exponentiation
- **Square Root**: Use binary search
- **Combinatorics**: Use DP or mathematical formulas
- **String Math**: Simulate arithmetic operations

### **Optimization Tips**

- **Avoid Overflow**: Use appropriate data types
- **Fast Algorithms**: Use binary exponentiation, binary search
- **Mathematical Formulas**: Use known formulas when possible
- **Modular Arithmetic**: Handle large numbers with modulo
