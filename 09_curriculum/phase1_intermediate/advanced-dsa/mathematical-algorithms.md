# Mathematical Algorithms

## Overview

This module covers advanced mathematical algorithms including number theory, combinatorics, probability, and optimization algorithms. These algorithms are essential for competitive programming, cryptography, and scientific computing.

## Table of Contents

1. [Number Theory](#number-theory)
2. [Combinatorics](#combinatorics)
3. [Probability and Statistics](#probability-and-statistics)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Geometric Algorithms](#geometric-algorithms)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Number Theory

### Greatest Common Divisor (GCD)

#### Theory
The GCD of two integers is the largest positive integer that divides both numbers without leaving a remainder.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

// Euclidean Algorithm
func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

// Extended Euclidean Algorithm
func extendedGCD(a, b int) (int, int, int) {
    if a == 0 {
        return b, 0, 1
    }
    
    gcd, x1, y1 := extendedGCD(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return gcd, x, y
}

// Modular Inverse
func modularInverse(a, m int) int {
    gcd, x, _ := extendedGCD(a, m)
    if gcd != 1 {
        return -1 // Inverse doesn't exist
    }
    
    return ((x % m) + m) % m
}

func main() {
    a, b := 56, 15
    result := gcd(a, b)
    fmt.Printf("GCD(%d, %d) = %d\n", a, b, result)
    
    // Extended GCD
    gcd, x, y := extendedGCD(a, b)
    fmt.Printf("Extended GCD: %d*%d + %d*%d = %d\n", a, x, b, y, a*x+b*y)
    
    // Modular inverse
    a, m := 3, 11
    inv := modularInverse(a, m)
    if inv != -1 {
        fmt.Printf("Modular inverse of %d mod %d is %d\n", a, m, inv)
    } else {
        fmt.Printf("Modular inverse of %d mod %d doesn't exist\n", a, m)
    }
}
```

##### Node.js Implementation

```javascript
// Euclidean Algorithm
function gcd(a, b) {
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    return a;
}

// Extended Euclidean Algorithm
function extendedGCD(a, b) {
    if (a === 0) {
        return [b, 0, 1];
    }
    
    const [gcd, x1, y1] = extendedGCD(b % a, a);
    const x = y1 - Math.floor(b / a) * x1;
    const y = x1;
    
    return [gcd, x, y];
}

// Modular Inverse
function modularInverse(a, m) {
    const [gcd, x] = extendedGCD(a, m);
    if (gcd !== 1) {
        return -1; // Inverse doesn't exist
    }
    
    return ((x % m) + m) % m;
}

// Example usage
const a = 56, b = 15;
const result = gcd(a, b);
console.log(`GCD(${a}, ${b}) = ${result}`);

// Extended GCD
const [gcd, x, y] = extendedGCD(a, b);
console.log(`Extended GCD: ${a}*${x} + ${b}*${y} = ${a * x + b * y}`);

// Modular inverse
const a2 = 3, m = 11;
const inv = modularInverse(a2, m);
if (inv !== -1) {
    console.log(`Modular inverse of ${a2} mod ${m} is ${inv}`);
} else {
    console.log(`Modular inverse of ${a2} mod ${m} doesn't exist`);
}
```

### Prime Number Algorithms

#### Sieve of Eratosthenes

##### Golang Implementation

```go
package main

import "fmt"

func sieveOfEratosthenes(n int) []int {
    isPrime := make([]bool, n+1)
    for i := 2; i <= n; i++ {
        isPrime[i] = true
    }
    
    for p := 2; p*p <= n; p++ {
        if isPrime[p] {
            for i := p * p; i <= n; i += p {
                isPrime[i] = false
            }
        }
    }
    
    var primes []int
    for i := 2; i <= n; i++ {
        if isPrime[i] {
            primes = append(primes, i)
        }
    }
    
    return primes
}

func main() {
    n := 30
    primes := sieveOfEratosthenes(n)
    fmt.Printf("Primes up to %d: %v\n", n, primes)
}
```

#### Fast Exponentiation

##### Golang Implementation

```go
package main

import "fmt"

func fastExponentiation(base, exponent, mod int) int {
    result := 1
    base = base % mod
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % mod
        }
        exponent = exponent >> 1
        base = (base * base) % mod
    }
    
    return result
}

func main() {
    base := 2
    exponent := 1000000
    mod := 1000000007
    
    result := fastExponentiation(base, exponent, mod)
    fmt.Printf("%d^%d mod %d = %d\n", base, exponent, mod, result)
}
```

## Combinatorics

### Factorial and Permutations

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

func factorialMod(n, mod int) int {
    result := 1
    for i := 2; i <= n; i++ {
        result = (result * i) % mod
    }
    return result
}

func permutations(n, r int) int {
    if r > n {
        return 0
    }
    result := 1
    for i := 0; i < r; i++ {
        result *= (n - i)
    }
    return result
}

func combinations(n, r int) int {
    if r > n || r < 0 {
        return 0
    }
    if r > n-r {
        r = n - r
    }
    
    result := 1
    for i := 0; i < r; i++ {
        result = result * (n - i) / (i + 1)
    }
    return result
}

func main() {
    n := 5
    r := 3
    
    fmt.Printf("Factorial of %d: %d\n", n, factorial(n))
    fmt.Printf("Permutations of %d things taken %d at a time: %d\n", n, r, permutations(n, r))
    fmt.Printf("Combinations of %d things taken %d at a time: %d\n", n, r, combinations(n, r))
}
```

### Catalan Numbers

#### Theory
Catalan numbers are a sequence of natural numbers that occur in various counting problems.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func catalanNumber(n int) int {
    if n <= 1 {
        return 1
    }
    
    result := 0
    for i := 0; i < n; i++ {
        result += catalanNumber(i) * catalanNumber(n-i-1)
    }
    
    return result
}

func catalanNumberDP(n int) int {
    if n <= 1 {
        return 1
    }
    
    dp := make([]int, n+1)
    dp[0] = 1
    dp[1] = 1
    
    for i := 2; i <= n; i++ {
        for j := 0; j < i; j++ {
            dp[i] += dp[j] * dp[i-j-1]
        }
    }
    
    return dp[n]
}

func main() {
    n := 10
    result := catalanNumberDP(n)
    fmt.Printf("Catalan number C(%d) = %d\n", n, result)
}
```

## Probability and Statistics

### Monte Carlo Methods

#### Theory
Monte Carlo methods use random sampling to solve mathematical problems.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
)

func monteCarloPi(n int) float64 {
    inside := 0
    rand.Seed(time.Now().UnixNano())
    
    for i := 0; i < n; i++ {
        x := rand.Float64()
        y := rand.Float64()
        if x*x+y*y <= 1 {
            inside++
        }
    }
    
    return 4.0 * float64(inside) / float64(n)
}

func main() {
    n := 1000000
    pi := monteCarloPi(n)
    fmt.Printf("Estimated π with %d samples: %.6f\n", n, pi)
    fmt.Printf("Actual π: %.6f\n", math.Pi)
    fmt.Printf("Error: %.6f\n", math.Abs(pi-math.Pi))
}
```

### Expected Value Calculations

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func expectedDiceRolls(target int) float64 {
    memo := make([]float64, target+1)
    for i := range memo {
        memo[i] = -1
    }
    
    var dp func(int) float64
    dp = func(sum int) float64 {
        if sum >= target {
            return 0
        }
        
        if memo[sum] != -1 {
            return memo[sum]
        }
        
        result := 1.0 // Current roll
        for face := 1; face <= 6; face++ {
            result += dp(sum + face) / 6.0
        }
        
        memo[sum] = result
        return result
    }
    
    return dp(0)
}

func main() {
    target := 10
    expected := expectedDiceRolls(target)
    fmt.Printf("Expected dice rolls to reach sum %d: %.4f\n", target, expected)
}
```

## Optimization Algorithms

### Binary Search

#### Theory
Binary search efficiently finds a target value in a sorted array.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

func binarySearchFirst(arr []int, target int) int {
    left, right := 0, len(arr)-1
    result := -1
    
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            result = mid
            right = mid - 1
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

func main() {
    arr := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    target := 5
    
    index := binarySearch(arr, target)
    if index != -1 {
        fmt.Printf("Found %d at index %d\n", target, index)
    } else {
        fmt.Printf("%d not found\n", target)
    }
}
```

### Ternary Search

#### Theory
Ternary search finds the maximum or minimum of a unimodal function.

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func ternarySearchMin(f func(float64) float64, left, right, eps float64) float64 {
    for right-left > eps {
        mid1 := left + (right-left)/3
        mid2 := right - (right-left)/3
        
        if f(mid1) > f(mid2) {
            left = mid1
        } else {
            right = mid2
        }
    }
    
    return (left + right) / 2
}

func main() {
    // Example: Find minimum of f(x) = x^2 - 4x + 3
    f := func(x float64) float64 {
        return x*x - 4*x + 3
    }
    
    left, right := -10.0, 10.0
    eps := 1e-6
    
    minX := ternarySearchMin(f, left, right, eps)
    minY := f(minX)
    
    fmt.Printf("Minimum at x = %.6f, f(x) = %.6f\n", minX, minY)
}
```

## Geometric Algorithms

### Convex Hull

#### Theory
The convex hull of a set of points is the smallest convex polygon that contains all the points.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type Point struct {
    x, y int
}

func crossProduct(o, a, b Point) int {
    return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x)
}

func convexHull(points []Point) []Point {
    n := len(points)
    if n < 3 {
        return points
    }
    
    // Sort points by x-coordinate, then by y-coordinate
    sort.Slice(points, func(i, j int) bool {
        if points[i].x == points[j].x {
            return points[i].y < points[j].y
        }
        return points[i].x < points[j].x
    })
    
    // Build lower hull
    lower := []Point{}
    for i := 0; i < n; i++ {
        for len(lower) >= 2 && crossProduct(lower[len(lower)-2], lower[len(lower)-1], points[i]) <= 0 {
            lower = lower[:len(lower)-1]
        }
        lower = append(lower, points[i])
    }
    
    // Build upper hull
    upper := []Point{}
    for i := n - 1; i >= 0; i-- {
        for len(upper) >= 2 && crossProduct(upper[len(upper)-2], upper[len(upper)-1], points[i]) <= 0 {
            upper = upper[:len(upper)-1]
        }
        upper = append(upper, points[i])
    }
    
    // Remove duplicates
    hull := append(lower[:len(lower)-1], upper[:len(upper)-1]...)
    return hull
}

func main() {
    points := []Point{
        {0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 0},
        {2, 0}, {1, 2}, {3, 2}, {2, 3},
    }
    
    hull := convexHull(points)
    fmt.Println("Convex Hull points:")
    for _, p := range hull {
        fmt.Printf("(%d, %d)\n", p.x, p.y)
    }
}
```

## Follow-up Questions

### 1. Number Theory
**Q: When would you use the extended Euclidean algorithm vs the basic Euclidean algorithm?**
A: Use the basic Euclidean algorithm when you only need the GCD. Use the extended version when you need to find coefficients x and y such that ax + by = gcd(a, b), which is useful for modular inverses and solving linear Diophantine equations.

### 2. Combinatorics
**Q: How do you handle large factorials and combinations that overflow standard integer types?**
A: Use modular arithmetic with a large prime modulus, precompute factorials and inverse factorials, or use the multiplicative formula for combinations to avoid computing large factorials directly.

### 3. Optimization
**Q: What's the difference between binary search and ternary search?**
A: Binary search works on sorted arrays and finds exact values. Ternary search works on unimodal functions and finds the maximum or minimum value. Ternary search divides the search space into three parts instead of two.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| GCD (Euclidean) | O(log min(a,b)) | O(1) | Iterative version |
| Extended GCD | O(log min(a,b)) | O(log min(a,b)) | Recursive version |
| Sieve of Eratosthenes | O(n log log n) | O(n) | Prime generation |
| Fast Exponentiation | O(log n) | O(1) | Modular exponentiation |
| Binary Search | O(log n) | O(1) | Sorted array |
| Ternary Search | O(log n) | O(1) | Unimodal function |
| Convex Hull | O(n log n) | O(n) | Graham scan |

## Applications

1. **Number Theory**: Cryptography, RSA encryption, hash functions
2. **Combinatorics**: Counting problems, probability calculations
3. **Probability**: Monte Carlo simulations, statistical analysis
4. **Optimization**: Search algorithms, numerical methods
5. **Geometry**: Computer graphics, collision detection, pathfinding

---

**Next**: [Applications](./applications.md) | **Previous**: [Advanced DSA](../README.md) | **Up**: [Advanced DSA](../README.md)
