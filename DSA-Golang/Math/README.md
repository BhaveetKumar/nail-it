# Math Pattern

> **Master mathematical algorithms and number theory with Go implementations**

## 📋 Problems

### **Number Theory**
- [Pow(x, n)](./Pow.md) - Calculate x raised to power n
- [Sqrt(x)](./Sqrt.md) - Calculate square root
- [Valid Perfect Square](./ValidPerfectSquare.md) - Check if number is perfect square
- [Power of Three](./PowerOfThree.md) - Check if number is power of 3
- [Roman to Integer](./RomanToInteger.md) - Convert Roman numerals to integer

### **Combinatorics**
- [Pascal's Triangle](./PascalsTriangle.md) - Generate Pascal's triangle
- [Pascal's Triangle II](./PascalsTriangleII.md) - Get specific row
- [Unique Paths](./UniquePaths.md) - Count paths in grid
- [Unique Paths II](./UniquePathsII.md) - Count paths with obstacles
- [Climbing Stairs](./ClimbingStairs.md) - Count ways to climb stairs

### **Arithmetic**
- [Add Binary](./AddBinary.md) - Add two binary strings
- [Multiply Strings](./MultiplyStrings.md) - Multiply two strings
- [Plus One](./PlusOne.md) - Add one to number represented as array
- [Factorial Trailing Zeroes](./FactorialTrailingZeroes.md) - Count trailing zeros in factorial
- [Excel Sheet Column Title](./ExcelSheetColumnTitle.md) - Convert number to Excel column

---

## 🎯 Key Concepts

### **Mathematical Operations**
- **Exponentiation**: Fast power calculation
- **Square Root**: Binary search approach
- **Modular Arithmetic**: Handle large numbers
- **Combinatorics**: Permutations and combinations

### **Common Patterns**
- **Binary Search**: For square root, power calculations
- **Dynamic Programming**: For counting problems
- **String Manipulation**: For arithmetic on large numbers
- **Number Theory**: Prime numbers, GCD, LCM

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
