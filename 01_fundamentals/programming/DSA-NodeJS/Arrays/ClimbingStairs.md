# 🪜 Climbing Stairs - LeetCode Problem 70

## Problem Statement

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

## Examples

```javascript
// Example 1
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

// Example 2
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
```

## Approach

### Dynamic Programming (Fibonacci Pattern)

This problem follows the **Fibonacci sequence** pattern:
- `ways(n) = ways(n-1) + ways(n-2)`
- Base cases: `ways(1) = 1`, `ways(2) = 2`

### Time Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(1) - Only using two variables

## Solution

### Approach 1: Dynamic Programming (Space Optimized)

```javascript
/**
 * @param {number} n
 * @return {number}
 */
function climbStairs(n) {
  if (n <= 2) return n;
  
  let prev2 = 1; // ways to reach step 1
  let prev1 = 2; // ways to reach step 2
  
  for (let i = 3; i <= n; i++) {
    const current = prev1 + prev2;
    prev2 = prev1;
    prev1 = current;
  }
  
  return prev1;
}
```

### Approach 2: Dynamic Programming with Array

```javascript
/**
 * DP with array approach
 * @param {number} n
 * @return {number}
 */
function climbStairsDP(n) {
  if (n <= 2) return n;
  
  const dp = new Array(n + 1);
  dp[1] = 1;
  dp[2] = 2;
  
  for (let i = 3; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  
  return dp[n];
}
```

### Approach 3: Recursive with Memoization

```javascript
/**
 * Recursive with memoization
 * @param {number} n
 * @return {number}
 */
function climbStairsMemo(n) {
  const memo = new Map();
  
  function helper(n) {
    if (n <= 2) return n;
    if (memo.has(n)) return memo.get(n);
    
    const result = helper(n - 1) + helper(n - 2);
    memo.set(n, result);
    return result;
  }
  
  return helper(n);
}
```

### Approach 4: Matrix Exponentiation (Advanced)

```javascript
/**
 * Matrix exponentiation approach for O(log n) time complexity
 * @param {number} n
 * @return {number}
 */
function climbStairsMatrix(n) {
  if (n <= 2) return n;
  
  // Matrix representation of Fibonacci
  // [F(n+1)]   [1 1]^n   [F(1)]
  // [F(n)  ] = [1 0]   * [F(0)]
  
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
    } else {
      return matrixMultiply(matrix, matrixPower(matrix, power - 1));
    }
  }
  
  const baseMatrix = [[1, 1], [1, 0]];
  const resultMatrix = matrixPower(baseMatrix, n - 1);
  
  // F(n) = resultMatrix[0][0] * F(1) + resultMatrix[0][1] * F(0)
  return resultMatrix[0][0] * 1 + resultMatrix[0][1] * 1;
}
```

### Approach 5: Mathematical Formula (Binet's Formula)

```javascript
/**
 * Binet's formula for Fibonacci numbers
 * @param {number} n
 * @return {number}
 */
function climbStairsBinet(n) {
  if (n <= 2) return n;
  
  const sqrt5 = Math.sqrt(5);
  const phi = (1 + sqrt5) / 2; // Golden ratio
  const psi = (1 - sqrt5) / 2;
  
  // F(n) = (phi^n - psi^n) / sqrt(5)
  // For climbing stairs, we need F(n+1)
  const result = Math.round((Math.pow(phi, n + 1) - Math.pow(psi, n + 1)) / sqrt5);
  return result;
}
```

## Extended Problem: Climbing Stairs with Variable Steps

```javascript
/**
 * Extended problem: Can climb 1, 2, or 3 steps at a time
 * @param {number} n
 * @return {number}
 */
function climbStairsExtended(n) {
  if (n <= 2) return n;
  if (n === 3) return 4; // 1+1+1, 1+2, 2+1, 3
  
  let prev3 = 1; // ways to reach step 1
  let prev2 = 2; // ways to reach step 2
  let prev1 = 4; // ways to reach step 3
  
  for (let i = 4; i <= n; i++) {
    const current = prev1 + prev2 + prev3;
    prev3 = prev2;
    prev2 = prev1;
    prev1 = current;
  }
  
  return prev1;
}

/**
 * Generic solution for any number of allowed steps
 * @param {number} n - total steps
 * @param {number[]} allowedSteps - array of allowed step sizes
 * @return {number}
 */
function climbStairsGeneric(n, allowedSteps) {
  const dp = new Array(n + 1).fill(0);
  dp[0] = 1; // One way to stay at ground level
  
  for (let i = 1; i <= n; i++) {
    for (const step of allowedSteps) {
      if (i >= step) {
        dp[i] += dp[i - step];
      }
    }
  }
  
  return dp[n];
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Climbing Stairs Test Cases ===");

// Test 1
console.log("Test 1: n = 2");
console.log("Output:", climbStairs(2));
console.log("Expected: 2");
console.log();

// Test 2
console.log("Test 2: n = 3");
console.log("Output:", climbStairs(3));
console.log("Expected: 3");
console.log();

// Test 3
console.log("Test 3: n = 5");
console.log("Output:", climbStairs(5));
console.log("Expected: 8");
console.log();

// Test 4
console.log("Test 4: n = 10");
console.log("Output:", climbStairs(10));
console.log("Expected: 89");
console.log();

// Performance comparison
console.log("=== Performance Comparison ===");
const testN = 40;

console.log(`Testing with n = ${testN}:`);

// DP approach
let start = performance.now();
let result1 = climbStairs(testN);
let end = performance.now();
console.log(`DP (Space Optimized): ${result1} - Time: ${end - start}ms`);

// DP with array
start = performance.now();
let result2 = climbStairsDP(testN);
end = performance.now();
console.log(`DP (Array): ${result2} - Time: ${end - start}ms`);

// Memoization
start = performance.now();
let result3 = climbStairsMemo(testN);
end = performance.now();
console.log(`Memoization: ${result3} - Time: ${end - start}ms`);

// Matrix exponentiation
start = performance.now();
let result4 = climbStairsMatrix(testN);
end = performance.now();
console.log(`Matrix Exponentiation: ${result4} - Time: ${end - start}ms`);

// Binet's formula
start = performance.now();
let result5 = climbStairsBinet(testN);
end = performance.now();
console.log(`Binet's Formula: ${result5} - Time: ${end - start}ms`);

// Extended problem
console.log("\n=== Extended Problem ===");
console.log("Climbing with 1, 2, or 3 steps:");
for (let i = 1; i <= 10; i++) {
  console.log(`n = ${i}: ${climbStairsExtended(i)} ways`);
}

// Generic solution
console.log("\n=== Generic Solution ===");
console.log("Allowed steps: [1, 2, 3, 5]");
for (let i = 1; i <= 10; i++) {
  console.log(`n = ${i}: ${climbStairsGeneric(i, [1, 2, 3, 5])} ways`);
}
```

## Visualization

```javascript
/**
 * Visualize all possible paths for small n
 * @param {number} n
 * @return {string[]}
 */
function visualizePaths(n) {
  const paths = [];
  
  function generatePaths(current, path) {
    if (current === n) {
      paths.push(path.join(' + '));
      return;
    }
    
    if (current + 1 <= n) {
      generatePaths(current + 1, [...path, 1]);
    }
    
    if (current + 2 <= n) {
      generatePaths(current + 2, [...path, 2]);
    }
  }
  
  generatePaths(0, []);
  return paths;
}

// Example visualization
console.log("=== Path Visualization ===");
console.log("n = 4 paths:");
const paths = visualizePaths(4);
paths.forEach((path, index) => {
  console.log(`${index + 1}. ${path}`);
});
```

## Key Insights

1. **Fibonacci Pattern**: This problem is essentially asking for the (n+1)th Fibonacci number
2. **Dynamic Programming**: Optimal substructure and overlapping subproblems
3. **Space Optimization**: Only need to keep track of previous two values
4. **Multiple Approaches**: DP, recursion, matrix exponentiation, mathematical formula
5. **Extensibility**: Can be extended to any number of allowed step sizes

## Common Mistakes

1. **Not recognizing the Fibonacci pattern**
2. **Using recursion without memoization** (exponential time)
3. **Not handling base cases** properly
4. **Off-by-one errors** in indexing
5. **Not optimizing space** when using DP array

## Related Problems

- [Fibonacci Number](../DynamicProgramming/Fibonacci.md)
- [House Robber](../../../algorithms/DynamicProgramming/HouseRobber.md)
- [Min Cost Climbing Stairs](MinCostClimbingStairs.md)
- [Decode Ways](../../../algorithms/Strings/DecodeWays.md)

## Interview Tips

1. **Start with recursive solution** and identify the pattern
2. **Optimize with memoization** to avoid redundant calculations
3. **Convert to iterative DP** for better space efficiency
4. **Discuss time/space complexity** trade-offs
5. **Mention the Fibonacci connection** as it shows pattern recognition
6. **Consider edge cases** like n = 0, n = 1
7. **Discuss extensibility** to different step sizes
