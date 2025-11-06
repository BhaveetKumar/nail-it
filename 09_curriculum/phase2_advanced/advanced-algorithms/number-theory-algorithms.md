---
# Auto-generated front matter
Title: Number-Theory-Algorithms
LastUpdated: 2025-11-06T20:45:58.428587
Tags: []
Status: draft
---

# Number Theory Algorithms

## Overview

This module covers number theory algorithms including prime number generation, modular arithmetic, greatest common divisor, and cryptographic applications. These concepts are essential for cryptography, computer security, and mathematical computations.

## Table of Contents

1. [Prime Number Algorithms](#prime-number-algorithms)
2. [Modular Arithmetic](#modular-arithmetic)
3. [Greatest Common Divisor](#greatest-common-divisor)
4. [Cryptographic Applications](#cryptographic-applications)
5. [Applications](#applications)
6. [Complexity Analysis](#complexity-analysis)
7. [Follow-up Questions](#follow-up-questions)

## Prime Number Algorithms

### Theory

Prime numbers are fundamental in number theory and cryptography. Efficient algorithms for prime generation, primality testing, and factorization are crucial for many applications.

### Prime Number Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
    "math/big"
    "math/rand"
    "time"
)

type PrimeGenerator struct {
    primes []int64
    mutex  sync.RWMutex
}

func NewPrimeGenerator() *PrimeGenerator {
    return &PrimeGenerator{
        primes: []int64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29},
    }
}

func (pg *PrimeGenerator) IsPrime(n int64) bool {
    if n < 2 {
        return false
    }
    if n == 2 {
        return true
    }
    if n%2 == 0 {
        return false
    }
    
    // Check against known primes first
    pg.mutex.RLock()
    for _, p := range pg.primes {
        if p*p > n {
            break
        }
        if n%p == 0 {
            pg.mutex.RUnlock()
            return false
        }
    }
    pg.mutex.RUnlock()
    
    // Trial division for larger numbers
    for i := int64(3); i*i <= n; i += 2 {
        if n%i == 0 {
            return false
        }
    }
    
    return true
}

func (pg *PrimeGenerator) SieveOfEratosthenes(limit int64) []int64 {
    if limit < 2 {
        return []int64{}
    }
    
    // Create boolean array
    isPrime := make([]bool, limit+1)
    for i := int64(2); i <= limit; i++ {
        isPrime[i] = true
    }
    
    // Sieve
    for i := int64(2); i*i <= limit; i++ {
        if isPrime[i] {
            for j := i * i; j <= limit; j += i {
                isPrime[j] = false
            }
        }
    }
    
    // Collect primes
    var primes []int64
    for i := int64(2); i <= limit; i++ {
        if isPrime[i] {
            primes = append(primes, i)
        }
    }
    
    return primes
}

func (pg *PrimeGenerator) GeneratePrimes(count int) []int64 {
    if count <= 0 {
        return []int64{}
    }
    
    var primes []int64
    candidate := int64(2)
    
    for len(primes) < count {
        if pg.IsPrime(candidate) {
            primes = append(primes, candidate)
        }
        candidate++
    }
    
    return primes
}

func (pg *PrimeGenerator) NextPrime(n int64) int64 {
    candidate := n + 1
    if candidate%2 == 0 {
        candidate++
    }
    
    for !pg.IsPrime(candidate) {
        candidate += 2
    }
    
    return candidate
}

func (pg *PrimeGenerator) PreviousPrime(n int64) int64 {
    if n <= 2 {
        return 0
    }
    
    candidate := n - 1
    if candidate%2 == 0 {
        candidate--
    }
    
    for candidate > 2 && !pg.IsPrime(candidate) {
        candidate -= 2
    }
    
    if candidate == 2 {
        return 2
    }
    
    return candidate
}

func (pg *PrimeGenerator) MillerRabinTest(n int64, k int) bool {
    if n < 2 {
        return false
    }
    if n == 2 || n == 3 {
        return true
    }
    if n%2 == 0 {
        return false
    }
    
    // Write n-1 as d * 2^r
    d := n - 1
    r := 0
    for d%2 == 0 {
        d /= 2
        r++
    }
    
    // Witness loop
    for i := 0; i < k; i++ {
        a := rand.Int63n(n-3) + 2
        x := pg.modularExponentiation(a, d, n)
        
        if x == 1 || x == n-1 {
            continue
        }
        
        for j := 0; j < r-1; j++ {
            x = (x * x) % n
            if x == n-1 {
                break
            }
        }
        
        if x != n-1 {
            return false
        }
    }
    
    return true
}

func (pg *PrimeGenerator) modularExponentiation(base, exponent, modulus int64) int64 {
    result := int64(1)
    base = base % modulus
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % modulus
        }
        exponent = exponent >> 1
        base = (base * base) % modulus
    }
    
    return result
}

func (pg *PrimeGenerator) Factorize(n int64) map[int64]int {
    factors := make(map[int64]int)
    
    if n < 2 {
        return factors
    }
    
    // Check for 2
    for n%2 == 0 {
        factors[2]++
        n /= 2
    }
    
    // Check for odd factors
    for i := int64(3); i*i <= n; i += 2 {
        for n%i == 0 {
            factors[i]++
            n /= i
        }
    }
    
    // If n is still greater than 1, it's a prime factor
    if n > 1 {
        factors[n]++
    }
    
    return factors
}

func main() {
    pg := NewPrimeGenerator()
    
    fmt.Println("Prime Number Algorithms Demo:")
    
    // Test primality
    testNumbers := []int64{17, 25, 29, 97, 100}
    for _, n := range testNumbers {
        isPrime := pg.IsPrime(n)
        fmt.Printf("%d is prime: %v\n", n, isPrime)
    }
    
    // Sieve of Eratosthenes
    primes := pg.SieveOfEratosthenes(50)
    fmt.Printf("Primes up to 50: %v\n", primes)
    
    // Generate first 10 primes
    first10 := pg.GeneratePrimes(10)
    fmt.Printf("First 10 primes: %v\n", first10)
    
    // Next and previous prime
    next := pg.NextPrime(20)
    prev := pg.PreviousPrime(20)
    fmt.Printf("Next prime after 20: %d\n", next)
    fmt.Printf("Previous prime before 20: %d\n", prev)
    
    // Miller-Rabin test
    rand.Seed(time.Now().UnixNano())
    isProbablePrime := pg.MillerRabinTest(97, 5)
    fmt.Printf("97 is probably prime (Miller-Rabin): %v\n", isProbablePrime)
    
    // Factorization
    factors := pg.Factorize(60)
    fmt.Printf("Factors of 60: %v\n", factors)
}
```

## Modular Arithmetic

### Theory

Modular arithmetic is fundamental in cryptography and computer science. It involves operations on integers with a fixed modulus, providing the foundation for many cryptographic algorithms.

### Modular Arithmetic Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type ModularArithmetic struct {
    Modulus int64
}

func NewModularArithmetic(modulus int64) *ModularArithmetic {
    return &ModularArithmetic{
        Modulus: modulus,
    }
}

func (ma *ModularArithmetic) Add(a, b int64) int64 {
    return ((a % ma.Modulus) + (b % ma.Modulus)) % ma.Modulus
}

func (ma *ModularArithmetic) Subtract(a, b int64) int64 {
    result := (a % ma.Modulus) - (b % ma.Modulus)
    if result < 0 {
        result += ma.Modulus
    }
    return result
}

func (ma *ModularArithmetic) Multiply(a, b int64) int64 {
    return ((a % ma.Modulus) * (b % ma.Modulus)) % ma.Modulus
}

func (ma *ModularArithmetic) Power(base, exponent int64) int64 {
    if exponent < 0 {
        // For negative exponents, we need modular inverse
        inverse := ma.ModularInverse(base)
        if inverse == -1 {
            return -1 // No inverse exists
        }
        return ma.Power(inverse, -exponent)
    }
    
    result := int64(1)
    base = base % ma.Modulus
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % ma.Modulus
        }
        exponent = exponent >> 1
        base = (base * base) % ma.Modulus
    }
    
    return result
}

func (ma *ModularArithmetic) ModularInverse(a int64) int64 {
    // Extended Euclidean Algorithm
    g, x, _ := ma.ExtendedGCD(a, ma.Modulus)
    if g != 1 {
        return -1 // No inverse exists
    }
    
    return (x%ma.Modulus + ma.Modulus) % ma.Modulus
}

func (ma *ModularArithmetic) ExtendedGCD(a, b int64) (int64, int64, int64) {
    if a == 0 {
        return b, 0, 1
    }
    
    g, x1, y1 := ma.ExtendedGCD(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return g, x, y
}

func (ma *ModularArithmetic) GCD(a, b int64) int64 {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func (ma *ModularArithmetic) LCM(a, b int64) int64 {
    return (a * b) / ma.GCD(a, b)
}

func (ma *ModularArithmetic) ChineseRemainderTheorem(remainders, moduli []int64) int64 {
    if len(remainders) != len(moduli) {
        return -1
    }
    
    n := len(remainders)
    if n == 0 {
        return 0
    }
    
    // Calculate product of all moduli
    product := int64(1)
    for _, m := range moduli {
        product *= m
    }
    
    result := int64(0)
    
    for i := 0; i < n; i++ {
        // Calculate Mi = product / moduli[i]
        Mi := product / moduli[i]
        
        // Calculate modular inverse of Mi mod moduli[i]
        MiInv := ma.ModularInverse(Mi)
        if MiInv == -1 {
            return -1 // No solution exists
        }
        
        result = (result + remainders[i]*Mi*MiInv) % product
    }
    
    return result
}

func (ma *ModularArithmetic) IsQuadraticResidue(a int64) bool {
    if a == 0 {
        return true
    }
    
    // Check if a^((p-1)/2) ≡ 1 (mod p)
    exponent := (ma.Modulus - 1) / 2
    result := ma.Power(a, exponent)
    return result == 1
}

func (ma *ModularArithmetic) TonelliShanks(n int64) int64 {
    if !ma.IsQuadraticResidue(n) {
        return -1 // No solution exists
    }
    
    if ma.Modulus == 2 {
        return n
    }
    
    if ma.Power(n, (ma.Modulus-1)/2) == ma.Modulus-1 {
        return -1 // No solution exists
    }
    
    // Find Q and S such that p-1 = Q * 2^S
    Q := ma.Modulus - 1
    S := int64(0)
    for Q%2 == 0 {
        Q /= 2
        S++
    }
    
    // Find a quadratic non-residue
    z := int64(2)
    for ma.IsQuadraticResidue(z) {
        z++
    }
    
    c := ma.Power(z, Q)
    x := ma.Power(n, (Q+1)/2)
    t := ma.Power(n, Q)
    m := S
    
    for t != 1 {
        // Find the least i such that t^(2^i) ≡ 1 (mod p)
        i := int64(1)
        for i < m && ma.Power(t, 1<<i) != 1 {
            i++
        }
        
        if i == m {
            return -1 // No solution exists
        }
        
        b := ma.Power(c, 1<<(m-i-1))
        x = ma.Multiply(x, b)
        t = ma.Multiply(t, ma.Multiply(b, b))
        c = ma.Multiply(b, b)
        m = i
    }
    
    return x
}

func (ma *ModularArithmetic) DiscreteLogarithm(base, result int64) int64 {
    // Baby-step giant-step algorithm
    m := int64(math.Ceil(math.Sqrt(float64(ma.Modulus))))
    
    // Baby steps
    table := make(map[int64]int64)
    baby := int64(1)
    for j := int64(0); j < m; j++ {
        table[baby] = j
        baby = ma.Multiply(baby, base)
    }
    
    // Giant steps
    giant := ma.Power(base, m)
    giant = ma.ModularInverse(giant)
    if giant == -1 {
        return -1
    }
    
    current := result
    for i := int64(0); i < m; i++ {
        if j, exists := table[current]; exists {
            return i*m + j
        }
        current = ma.Multiply(current, giant)
    }
    
    return -1 // No solution found
}

func main() {
    ma := NewModularArithmetic(17)
    
    fmt.Println("Modular Arithmetic Demo:")
    
    // Basic operations
    fmt.Printf("7 + 5 mod 17 = %d\n", ma.Add(7, 5))
    fmt.Printf("7 - 5 mod 17 = %d\n", ma.Subtract(7, 5))
    fmt.Printf("7 * 5 mod 17 = %d\n", ma.Multiply(7, 5))
    fmt.Printf("7^3 mod 17 = %d\n", ma.Power(7, 3))
    
    // Modular inverse
    inverse := ma.ModularInverse(7)
    fmt.Printf("7^(-1) mod 17 = %d\n", inverse)
    
    // GCD and LCM
    fmt.Printf("GCD(12, 8) = %d\n", ma.GCD(12, 8))
    fmt.Printf("LCM(12, 8) = %d\n", ma.LCM(12, 8))
    
    // Chinese Remainder Theorem
    remainders := []int64{2, 3, 2}
    moduli := []int64{3, 5, 7}
    crt := ma.ChineseRemainderTheorem(remainders, moduli)
    fmt.Printf("Chinese Remainder Theorem result: %d\n", crt)
    
    // Quadratic residue
    isQR := ma.IsQuadraticResidue(9)
    fmt.Printf("9 is quadratic residue mod 17: %v\n", isQR)
    
    // Tonelli-Shanks algorithm
    sqrt := ma.TonelliShanks(9)
    fmt.Printf("Square root of 9 mod 17: %d\n", sqrt)
    
    // Discrete logarithm
    dlog := ma.DiscreteLogarithm(3, 5)
    fmt.Printf("Discrete log of 5 base 3 mod 17: %d\n", dlog)
}
```

## Greatest Common Divisor

### Theory

The greatest common divisor (GCD) is the largest positive integer that divides two or more integers without remainder. The Euclidean algorithm is the most efficient method for computing GCD.

### GCD Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type GCDCalculator struct{}

func NewGCDCalculator() *GCDCalculator {
    return &GCDCalculator{}
}

func (gc *GCDCalculator) Euclidean(a, b int64) int64 {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func (gc *GCDCalculator) ExtendedEuclidean(a, b int64) (int64, int64, int64) {
    if a == 0 {
        return b, 0, 1
    }
    
    gcd, x1, y1 := gc.ExtendedEuclidean(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return gcd, x, y
}

func (gc *GCDCalculator) BinaryGCD(a, b int64) int64 {
    if a == 0 {
        return b
    }
    if b == 0 {
        return a
    }
    
    // Find common factors of 2
    shift := 0
    for (a|b)&1 == 0 {
        a >>= 1
        b >>= 1
        shift++
    }
    
    // Remove remaining factors of 2 from a
    for a&1 == 0 {
        a >>= 1
    }
    
    // Now a is odd
    for {
        // Remove remaining factors of 2 from b
        for b&1 == 0 {
            b >>= 1
        }
        
        // Now both a and b are odd
        if a > b {
            a, b = b, a
        }
        b -= a
        
        if b == 0 {
            break
        }
    }
    
    return a << shift
}

func (gc *GCDCalculator) LCM(a, b int64) int64 {
    if a == 0 || b == 0 {
        return 0
    }
    return (a * b) / gc.Euclidean(a, b)
}

func (gc *GCDCalculator) GCDArray(numbers []int64) int64 {
    if len(numbers) == 0 {
        return 0
    }
    
    result := numbers[0]
    for i := 1; i < len(numbers); i++ {
        result = gc.Euclidean(result, numbers[i])
        if result == 1 {
            break
        }
    }
    
    return result
}

func (gc *GCDCalculator) LCMArray(numbers []int64) int64 {
    if len(numbers) == 0 {
        return 0
    }
    
    result := numbers[0]
    for i := 1; i < len(numbers); i++ {
        result = gc.LCM(result, numbers[i])
    }
    
    return result
}

func (gc *GCDCalculator) BezoutCoefficients(a, b int64) (int64, int64, int64) {
    return gc.ExtendedEuclidean(a, b)
}

func (gc *GCDCalculator) ModularInverse(a, m int64) int64 {
    gcd, x, _ := gc.ExtendedEuclidean(a, m)
    if gcd != 1 {
        return -1 // No inverse exists
    }
    
    return (x%m + m) % m
}

func (gc *GCDCalculator) SolveLinearDiophantine(a, b, c int64) (int64, int64, bool) {
    gcd, x0, y0 := gc.ExtendedEuclidean(a, b)
    
    if c%gcd != 0 {
        return 0, 0, false // No solution exists
    }
    
    x := x0 * (c / gcd)
    y := y0 * (c / gcd)
    
    return x, y, true
}

func (gc *GCDCalculator) IsCoprime(a, b int64) bool {
    return gc.Euclidean(a, b) == 1
}

func (gc *GCDCalculator) Totient(n int64) int64 {
    result := n
    
    for i := int64(2); i*i <= n; i++ {
        if n%i == 0 {
            for n%i == 0 {
                n /= i
            }
            result -= result / i
        }
    }
    
    if n > 1 {
        result -= result / n
    }
    
    return result
}

func (gc *GCDCalculator) Carmichael(n int64) int64 {
    if n < 1 {
        return 0
    }
    
    // Factorize n
    factors := make(map[int64]int)
    temp := n
    
    for i := int64(2); i*i <= temp; i++ {
        for temp%i == 0 {
            factors[i]++
            temp /= i
        }
    }
    if temp > 1 {
        factors[temp]++
    }
    
    // Calculate Carmichael function
    result := int64(1)
    for p, e := range factors {
        if p == 2 {
            if e == 1 {
                result = gc.LCM(result, 1)
            } else if e == 2 {
                result = gc.LCM(result, 2)
            } else {
                result = gc.LCM(result, 1<<(e-2))
            }
        } else {
            result = gc.LCM(result, gc.Totient(int64(math.Pow(float64(p), float64(e)))))
        }
    }
    
    return result
}

func main() {
    gc := NewGCDCalculator()
    
    fmt.Println("Greatest Common Divisor Demo:")
    
    // Basic GCD
    a, b := int64(48), int64(18)
    gcd := gc.Euclidean(a, b)
    fmt.Printf("GCD(%d, %d) = %d\n", a, b, gcd)
    
    // Extended Euclidean
    gcd, x, y := gc.ExtendedEuclidean(a, b)
    fmt.Printf("Extended GCD(%d, %d) = %d, x = %d, y = %d\n", a, b, gcd, x, y)
    
    // Binary GCD
    binaryGCD := gc.BinaryGCD(a, b)
    fmt.Printf("Binary GCD(%d, %d) = %d\n", a, b, binaryGCD)
    
    // LCM
    lcm := gc.LCM(a, b)
    fmt.Printf("LCM(%d, %d) = %d\n", a, b, lcm)
    
    // GCD of array
    numbers := []int64{12, 18, 24, 30}
    arrayGCD := gc.GCDArray(numbers)
    fmt.Printf("GCD of %v = %d\n", numbers, arrayGCD)
    
    // LCM of array
    arrayLCM := gc.LCMArray(numbers)
    fmt.Printf("LCM of %v = %d\n", numbers, arrayLCM)
    
    // Modular inverse
    inverse := gc.ModularInverse(7, 17)
    fmt.Printf("7^(-1) mod 17 = %d\n", inverse)
    
    // Linear Diophantine equation
    x, y, solved := gc.SolveLinearDiophantine(3, 5, 7)
    if solved {
        fmt.Printf("Solution to 3x + 5y = 7: x = %d, y = %d\n", x, y)
    } else {
        fmt.Println("No solution to 3x + 5y = 7")
    }
    
    // Coprime check
    coprime := gc.IsCoprime(15, 28)
    fmt.Printf("15 and 28 are coprime: %v\n", coprime)
    
    // Euler's totient function
    totient := gc.Totient(12)
    fmt.Printf("φ(12) = %d\n", totient)
    
    // Carmichael function
    carmichael := gc.Carmichael(12)
    fmt.Printf("λ(12) = %d\n", carmichael)
}
```

## Follow-up Questions

### 1. Prime Number Algorithms
**Q: What is the difference between deterministic and probabilistic primality tests?**
A: Deterministic tests (like trial division) always give the correct answer but may be slow. Probabilistic tests (like Miller-Rabin) are faster but have a small chance of error.

### 2. Modular Arithmetic
**Q: When does a modular inverse exist?**
A: A modular inverse of a modulo m exists if and only if gcd(a, m) = 1, i.e., a and m are coprime.

### 3. Greatest Common Divisor
**Q: What is the time complexity of the Euclidean algorithm?**
A: The Euclidean algorithm has time complexity O(log min(a, b)), making it very efficient for computing GCD.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Trial Division | O(√n) | O(1) | Simple but slow |
| Sieve of Eratosthenes | O(n log log n) | O(n) | Efficient for range |
| Miller-Rabin | O(k log³n) | O(1) | Probabilistic |
| Euclidean GCD | O(log min(a,b)) | O(1) | Very efficient |
| Extended Euclidean | O(log min(a,b)) | O(log min(a,b)) | For coefficients |
| Binary GCD | O(log min(a,b)) | O(1) | Bit operations |

## Applications

1. **Prime Number Algorithms**: Cryptography, random number generation, hash functions
2. **Modular Arithmetic**: RSA encryption, elliptic curve cryptography, hash functions
3. **Greatest Common Divisor**: Fraction simplification, linear Diophantine equations
4. **Number Theory**: Computer security, cryptographic protocols, mathematical research

---

**Next**: [Graph Theory Algorithms](graph-theory-algorithms.md) | **Previous**: [Advanced Algorithms](README.md) | **Up**: [Advanced Algorithms](README.md)
