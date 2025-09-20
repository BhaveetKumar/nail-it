# Fibonacci Sequence

## Problem Statement

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1.

**Sequence:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

**Mathematical Definition:**
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Example:**
```
Input: n = 10
Output: 55
Explanation: F(10) = F(9) + F(8) = 34 + 21 = 55
```

## Approaches

### 1. Naive Recursive Approach
**Time Complexity:** O(2^n) - Exponential
**Space Complexity:** O(n) - Recursion stack

### 2. Memoization (Top-Down DP)
**Time Complexity:** O(n) - Each subproblem solved once
**Space Complexity:** O(n) - Memoization table + recursion stack

### 3. Tabulation (Bottom-Up DP)
**Time Complexity:** O(n) - Single pass through array
**Space Complexity:** O(n) - DP table

### 4. Space-Optimized DP
**Time Complexity:** O(n) - Single pass
**Space Complexity:** O(1) - Only store previous two values

## Solutions

### 1. Naive Recursive Solution
```javascript
/**
 * Calculate nth Fibonacci number using naive recursion
 * @param {number} n - Position in Fibonacci sequence
 * @return {number} - nth Fibonacci number
 */
function fibonacciNaive(n) {
    // Base cases
    if (n <= 1) {
        return n;
    }
    
    // Recursive case
    return fibonacciNaive(n - 1) + fibonacciNaive(n - 2);
}
```

### 2. Memoization Solution
```javascript
/**
 * Calculate nth Fibonacci number using memoization
 * @param {number} n - Position in Fibonacci sequence
 * @return {number} - nth Fibonacci number
 */
function fibonacciMemo(n, memo = {}) {
    // Base cases
    if (n <= 1) {
        return n;
    }
    
    // Check if already computed
    if (n in memo) {
        return memo[n];
    }
    
    // Compute and store result
    memo[n] = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
    return memo[n];
}

// Alternative implementation with array memoization
function fibonacciMemoArray(n) {
    const memo = Array(n + 1).fill(-1);
    
    function fib(n) {
        if (n <= 1) {
            return n;
        }
        
        if (memo[n] !== -1) {
            return memo[n];
        }
        
        memo[n] = fib(n - 1) + fib(n - 2);
        return memo[n];
    }
    
    return fib(n);
}
```

### 3. Tabulation Solution
```javascript
/**
 * Calculate nth Fibonacci number using tabulation
 * @param {number} n - Position in Fibonacci sequence
 * @return {number} - nth Fibonacci number
 */
function fibonacciTabulation(n) {
    if (n <= 1) {
        return n;
    }
    
    // Create DP table
    const dp = Array(n + 1).fill(0);
    dp[0] = 0;
    dp[1] = 1;
    
    // Fill table bottom-up
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}
```

### 4. Space-Optimized Solution
```javascript
/**
 * Calculate nth Fibonacci number with O(1) space
 * @param {number} n - Position in Fibonacci sequence
 * @return {number} - nth Fibonacci number
 */
function fibonacciOptimized(n) {
    if (n <= 1) {
        return n;
    }
    
    let prev2 = 0; // F(i-2)
    let prev1 = 1; // F(i-1)
    
    for (let i = 2; i <= n; i++) {
        const current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}

// Alternative implementation using array destructuring
function fibonacciOptimizedAlt(n) {
    if (n <= 1) {
        return n;
    }
    
    let [prev2, prev1] = [0, 1];
    
    for (let i = 2; i <= n; i++) {
        [prev2, prev1] = [prev1, prev1 + prev2];
    }
    
    return prev1;
}
```

## Dry Run

**Input:** n = 5

### Memoization Approach:
```
fibonacciMemo(5, {})

1. n = 5, not in memo
   memo[5] = fibonacciMemo(4, memo) + fibonacciMemo(3, memo)
   
2. fibonacciMemo(4, memo)
   n = 4, not in memo
   memo[4] = fibonacciMemo(3, memo) + fibonacciMemo(2, memo)
   
3. fibonacciMemo(3, memo)
   n = 3, not in memo
   memo[3] = fibonacciMemo(2, memo) + fibonacciMemo(1, memo)
   
4. fibonacciMemo(2, memo)
   n = 2, not in memo
   memo[2] = fibonacciMemo(1, memo) + fibonacciMemo(0, memo)
   
5. fibonacciMemo(1, memo)
   n = 1, return 1
   
6. fibonacciMemo(0, memo)
   n = 0, return 0
   
7. memo[2] = 1 + 0 = 1
8. memo[3] = 1 + 1 = 2
9. memo[4] = 2 + 1 = 3
10. memo[5] = 3 + 2 = 5

Result: 5
```

### Tabulation Approach:
```
fibonacciTabulation(5)

Initial: dp = [0, 1, 0, 0, 0, 0]

i = 2: dp[2] = dp[1] + dp[0] = 1 + 0 = 1
       dp = [0, 1, 1, 0, 0, 0]

i = 3: dp[3] = dp[2] + dp[1] = 1 + 1 = 2
       dp = [0, 1, 1, 2, 0, 0]

i = 4: dp[4] = dp[3] + dp[2] = 2 + 1 = 3
       dp = [0, 1, 1, 2, 3, 0]

i = 5: dp[5] = dp[4] + dp[3] = 3 + 2 = 5
       dp = [0, 1, 1, 2, 3, 5]

Result: dp[5] = 5
```

### Space-Optimized Approach:
```
fibonacciOptimized(5)

Initial: prev2 = 0, prev1 = 1

i = 2: current = 1 + 0 = 1
       prev2 = 1, prev1 = 1

i = 3: current = 1 + 1 = 2
       prev2 = 1, prev1 = 2

i = 4: current = 2 + 1 = 3
       prev2 = 2, prev1 = 3

i = 5: current = 3 + 2 = 5
       prev2 = 3, prev1 = 5

Result: prev1 = 5
```

## Complexity Analysis

| Approach | Time Complexity | Space Complexity |
|----------|----------------|------------------|
| Naive Recursive | O(2^n) | O(n) |
| Memoization | O(n) | O(n) |
| Tabulation | O(n) | O(n) |
| Space-Optimized | O(n) | O(1) |

## Test Cases

```javascript
// Test cases
console.log(fibonacciOptimized(0)); // 0
console.log(fibonacciOptimized(1)); // 1
console.log(fibonacciOptimized(2)); // 1
console.log(fibonacciOptimized(3)); // 2
console.log(fibonacciOptimized(4)); // 3
console.log(fibonacciOptimized(5)); // 5
console.log(fibonacciOptimized(10)); // 55
console.log(fibonacciOptimized(20)); // 6765

// Performance comparison
function performanceTest() {
    const n = 40;
    
    console.time('Naive');
    console.log(fibonacciNaive(n));
    console.timeEnd('Naive');
    
    console.time('Memoization');
    console.log(fibonacciMemo(n));
    console.timeEnd('Memoization');
    
    console.time('Tabulation');
    console.log(fibonacciTabulation(n));
    console.timeEnd('Tabulation');
    
    console.time('Optimized');
    console.log(fibonacciOptimized(n));
    console.timeEnd('Optimized');
}
```

## Advanced Variations

### 1. Fibonacci with Modulo
```javascript
function fibonacciModulo(n, mod) {
    if (n <= 1) return n % mod;
    
    let prev2 = 0, prev1 = 1;
    for (let i = 2; i <= n; i++) {
        const current = (prev1 + prev2) % mod;
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

### 2. Fibonacci Sequence Generator
```javascript
function* fibonacciGenerator() {
    let prev2 = 0, prev1 = 1;
    yield prev2;
    yield prev1;
    
    while (true) {
        const current = prev1 + prev2;
        yield current;
        prev2 = prev1;
        prev1 = current;
    }
}

// Usage
const fibGen = fibonacciGenerator();
console.log(fibGen.next().value); // 0
console.log(fibGen.next().value); // 1
console.log(fibGen.next().value); // 1
console.log(fibGen.next().value); // 2
```

### 3. Fibonacci with Matrix Exponentiation
```javascript
function fibonacciMatrix(n) {
    if (n <= 1) return n;
    
    function matrixMultiply(a, b) {
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ];
    }
    
    function matrixPower(matrix, power) {
        if (power === 1) return matrix;
        if (power % 2 === 0) {
            const half = matrixPower(matrix, power / 2);
            return matrixMultiply(half, half);
        }
        return matrixMultiply(matrix, matrixPower(matrix, power - 1));
    }
    
    const baseMatrix = [[1, 1], [1, 0]];
    const resultMatrix = matrixPower(baseMatrix, n - 1);
    return resultMatrix[0][0];
}
```

## Key Insights

1. **Overlapping Subproblems**: Same subproblems are solved multiple times
2. **Optimal Substructure**: Solution can be built from optimal solutions to subproblems
3. **Memoization**: Store results to avoid recomputation
4. **Space Optimization**: Only need previous two values for current computation
5. **Matrix Exponentiation**: Can achieve O(log n) time complexity for very large n

## Related Problems

- [Climbing Stairs](../../../algorithms/Arrays/ClimbingStairs.md) - Similar recurrence relation
- [House Robber](../../../algorithms/DynamicProgramming/HouseRobber.md) - DP with constraints
- [Min Cost Climbing Stairs](../Arrays/MinCostClimbingStairs.md) - Variation with costs
- [N-th Tribonacci Number](Tribonacci.md) - Three-term recurrence
