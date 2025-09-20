# Dynamic Programming Advanced

## Overview

This module covers advanced dynamic programming techniques including interval DP, bitmask DP, digit DP, probability DP, and optimization strategies. These techniques are essential for solving complex optimization problems efficiently.

## Table of Contents

1. [Interval DP](#interval-dp)
2. [Bitmask DP](#bitmask-dp)
3. [Digit DP](#digit-dp)
4. [Probability DP](#probability-dp)
5. [Space Optimization](#space-optimization)
6. [State Space Reduction](#state-space-reduction)
7. [DP on Trees](#dp-on-trees)
8. [DP on Graphs](#dp-on-graphs)
9. [Applications](#applications)
10. [Follow-up Questions](#follow-up-questions)

## Interval DP

### Theory

Interval DP solves problems by considering all possible intervals and building solutions from smaller intervals to larger ones. Common patterns include matrix chain multiplication, palindrome problems, and optimal binary search trees.

### Matrix Chain Multiplication

#### Problem
Given a sequence of matrices, find the most efficient way to multiply them together.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

func matrixChainOrder(p []int) (int, [][]int) {
    n := len(p) - 1
    m := make([][]int, n+1)
    s := make([][]int, n+1)
    
    for i := 0; i <= n; i++ {
        m[i] = make([]int, n+1)
        s[i] = make([]int, n+1)
    }
    
    // l is the chain length
    for l := 2; l <= n; l++ {
        for i := 1; i <= n-l+1; i++ {
            j := i + l - 1
            m[i][j] = math.MaxInt32
            
            for k := i; k < j; k++ {
                cost := m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]
                if cost < m[i][j] {
                    m[i][j] = cost
                    s[i][j] = k
                }
            }
        }
    }
    
    return m[1][n], s
}

func printOptimalParens(s [][]int, i, j int) {
    if i == j {
        fmt.Printf("A%d", i)
    } else {
        fmt.Print("(")
        printOptimalParens(s, i, s[i][j])
        printOptimalParens(s, s[i][j]+1, j)
        fmt.Print(")")
    }
}

func main() {
    // Matrix dimensions: A1(1x2), A2(2x3), A3(3x4), A4(4x5)
    p := []int{1, 2, 3, 4, 5}
    
    minCost, s := matrixChainOrder(p)
    fmt.Printf("Minimum number of multiplications: %d\n", minCost)
    fmt.Print("Optimal parenthesization: ")
    printOptimalParens(s, 1, len(p)-1)
    fmt.Println()
}
```

##### Node.js Implementation

```javascript
function matrixChainOrder(p) {
    const n = p.length - 1;
    const m = Array(n + 1).fill().map(() => Array(n + 1).fill(0));
    const s = Array(n + 1).fill().map(() => Array(n + 1).fill(0));
    
    // l is the chain length
    for (let l = 2; l <= n; l++) {
        for (let i = 1; i <= n - l + 1; i++) {
            const j = i + l - 1;
            m[i][j] = Infinity;
            
            for (let k = i; k < j; k++) {
                const cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (cost < m[i][j]) {
                    m[i][j] = cost;
                    s[i][j] = k;
                }
            }
        }
    }
    
    return { minCost: m[1][n], s };
}

function printOptimalParens(s, i, j) {
    if (i === j) {
        process.stdout.write(`A${i}`);
    } else {
        process.stdout.write('(');
        printOptimalParens(s, i, s[i][j]);
        printOptimalParens(s, s[i][j] + 1, j);
        process.stdout.write(')');
    }
}

// Example usage
const p = [1, 2, 3, 4, 5]; // Matrix dimensions
const { minCost, s } = matrixChainOrder(p);
console.log(`Minimum number of multiplications: ${minCost}`);
process.stdout.write('Optimal parenthesization: ');
printOptimalParens(s, 1, p.length - 1);
console.log();
```

### Palindrome Partitioning

#### Problem
Find the minimum number of cuts needed to partition a string into palindromes.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

func minCut(s string) int {
    n := len(s)
    isPalindrome := make([][]bool, n)
    for i := range isPalindrome {
        isPalindrome[i] = make([]bool, n)
    }
    
    // Check all possible palindromes
    for i := 0; i < n; i++ {
        for j := 0; j <= i; j++ {
            if s[i] == s[j] && (i-j <= 2 || isPalindrome[j+1][i-1]) {
                isPalindrome[j][i] = true
            }
        }
    }
    
    // DP to find minimum cuts
    cuts := make([]int, n+1)
    for i := 0; i <= n; i++ {
        cuts[i] = i - 1
    }
    
    for i := 0; i < n; i++ {
        for j := 0; j <= i; j++ {
            if isPalindrome[j][i] {
                if cuts[j]+1 < cuts[i+1] {
                    cuts[i+1] = cuts[j] + 1
                }
            }
        }
    }
    
    return cuts[n]
}

func main() {
    s := "aab"
    result := minCut(s)
    fmt.Printf("Minimum cuts for '%s': %d\n", s, result)
    
    s = "racecar"
    result = minCut(s)
    fmt.Printf("Minimum cuts for '%s': %d\n", s, result)
}
```

## Bitmask DP

### Theory

Bitmask DP uses bit manipulation to represent states, often used for problems involving subsets, permutations, or states that can be represented as binary numbers.

### Traveling Salesman Problem (TSP)

#### Problem
Find the shortest possible route that visits each city exactly once and returns to the origin city.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

func tsp(graph [][]int) int {
    n := len(graph)
    dp := make([][]int, 1<<n)
    for i := range dp {
        dp[i] = make([]int, n)
        for j := range dp[i] {
            dp[i][j] = math.MaxInt32
        }
    }
    
    // Base case: starting from city 0
    dp[1][0] = 0
    
    // Try all possible subsets
    for mask := 1; mask < (1 << n); mask++ {
        for u := 0; u < n; u++ {
            if (mask & (1 << u)) == 0 {
                continue
            }
            
            for v := 0; v < n; v++ {
                if (mask & (1 << v)) != 0 || graph[u][v] == 0 {
                    continue
                }
                
                newMask := mask | (1 << v)
                if dp[mask][u] + graph[u][v] < dp[newMask][v] {
                    dp[newMask][v] = dp[mask][u] + graph[u][v]
                }
            }
        }
    }
    
    // Find minimum cost to return to starting city
    result := math.MaxInt32
    fullMask := (1 << n) - 1
    for i := 1; i < n; i++ {
        if dp[fullMask][i] + graph[i][0] < result {
            result = dp[fullMask][i] + graph[i][0]
        }
    }
    
    return result
}

func main() {
    // Example graph (0-indexed)
    graph := [][]int{
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0},
    }
    
    result := tsp(graph)
    fmt.Printf("Minimum cost for TSP: %d\n", result)
}
```

##### Node.js Implementation

```javascript
function tsp(graph) {
    const n = graph.length;
    const dp = Array(1 << n).fill().map(() => Array(n).fill(Infinity));
    
    // Base case: starting from city 0
    dp[1][0] = 0;
    
    // Try all possible subsets
    for (let mask = 1; mask < (1 << n); mask++) {
        for (let u = 0; u < n; u++) {
            if ((mask & (1 << u)) === 0) continue;
            
            for (let v = 0; v < n; v++) {
                if ((mask & (1 << v)) !== 0 || graph[u][v] === 0) continue;
                
                const newMask = mask | (1 << v);
                if (dp[mask][u] + graph[u][v] < dp[newMask][v]) {
                    dp[newMask][v] = dp[mask][u] + graph[u][v];
                }
            }
        }
    }
    
    // Find minimum cost to return to starting city
    let result = Infinity;
    const fullMask = (1 << n) - 1;
    for (let i = 1; i < n; i++) {
        if (dp[fullMask][i] + graph[i][0] < result) {
            result = dp[fullMask][i] + graph[i][0];
        }
    }
    
    return result;
}

// Example usage
const graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
];

const result = tsp(graph);
console.log(`Minimum cost for TSP: ${result}`);
```

## Digit DP

### Theory

Digit DP solves problems involving digits of numbers, often counting numbers with certain properties or finding the k-th number with specific characteristics.

### Count Numbers with Digit Sum

#### Problem
Count numbers from 1 to N that have a digit sum equal to a given value.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "strconv"
)

func countNumbersWithDigitSum(n, targetSum int) int {
    s := strconv.Itoa(n)
    memo := make([][]int, len(s))
    for i := range memo {
        memo[i] = make([]int, targetSum+1)
        for j := range memo[i] {
            memo[i][j] = -1
        }
    }
    
    var dp func(int, int, bool) int
    dp = func(pos, sum int, tight bool) int {
        if pos == len(s) {
            if sum == targetSum {
                return 1
            }
            return 0
        }
        
        if !tight && memo[pos][sum] != -1 {
            return memo[pos][sum]
        }
        
        limit := 9
        if tight {
            limit = int(s[pos] - '0')
        }
        
        result := 0
        for digit := 0; digit <= limit; digit++ {
            newTight := tight && (digit == limit)
            newSum := sum + digit
            if newSum <= targetSum {
                result += dp(pos+1, newSum, newTight)
            }
        }
        
        if !tight {
            memo[pos][sum] = result
        }
        
        return result
    }
    
    return dp(0, 0, true)
}

func main() {
    n := 100
    targetSum := 5
    result := countNumbersWithDigitSum(n, targetSum)
    fmt.Printf("Numbers from 1 to %d with digit sum %d: %d\n", n, targetSum, result)
}
```

## Probability DP

### Theory

Probability DP solves problems involving probabilities and expected values, often using dynamic programming to compute complex probability distributions.

### Expected Value of Dice Rolls

#### Problem
Find the expected number of dice rolls needed to get a sum greater than or equal to a target value.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
)

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

## Space Optimization

### Theory

Space optimization techniques reduce the memory usage of DP solutions by reusing space or using rolling arrays.

### Fibonacci with Space Optimization

#### Implementations

##### Golang Implementation

```go
package main

import "fmt"

func fibonacciOptimized(n int) int {
    if n <= 1 {
        return n
    }
    
    prev2 := 0
    prev1 := 1
    
    for i := 2; i <= n; i++ {
        current := prev1 + prev2
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}

func main() {
    n := 10
    result := fibonacciOptimized(n)
    fmt.Printf("Fibonacci(%d) = %d\n", n, result)
}
```

## DP on Trees

### Theory

DP on trees solves problems by considering the tree structure and computing values bottom-up or top-down.

### Maximum Path Sum in Binary Tree

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func maxPathSum(root *TreeNode) int {
    maxSum := math.MinInt32
    
    var maxGain func(*TreeNode) int
    maxGain = func(node *TreeNode) int {
        if node == nil {
            return 0
        }
        
        leftGain := int(math.Max(0, float64(maxGain(node.Left))))
        rightGain := int(math.Max(0, float64(maxGain(node.Right))))
        
        currentMaxPath := node.Val + leftGain + rightGain
        maxSum = int(math.Max(float64(maxSum), float64(currentMaxPath)))
        
        return node.Val + int(math.Max(float64(leftGain), float64(rightGain)))
    }
    
    maxGain(root)
    return maxSum
}

func main() {
    // Example tree: [1,2,3]
    root := &TreeNode{
        Val: 1,
        Left: &TreeNode{Val: 2},
        Right: &TreeNode{Val: 3},
    }
    
    result := maxPathSum(root)
    fmt.Printf("Maximum path sum: %d\n", result)
}
```

## Follow-up Questions

### 1. DP Pattern Recognition
**Q: How do you identify when to use interval DP vs other DP patterns?**
A: Use interval DP when the problem involves ranges or intervals, and the optimal solution for a range depends on optimal solutions of smaller ranges. Look for problems involving matrix multiplication, palindromes, or optimal binary search trees.

### 2. Bitmask DP Applications
**Q: What types of problems are best solved with bitmask DP?**
A: Use bitmask DP for problems involving subsets, permutations, or states that can be represented as binary numbers. Common applications include TSP, assignment problems, and subset enumeration.

### 3. Space Optimization
**Q: When should you optimize space in DP solutions?**
A: Optimize space when the current solution uses O(n²) or more space but only needs the previous row/state, or when memory constraints are tight. Always consider readability vs optimization trade-offs.

## Complexity Analysis

| Technique | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Interval DP | O(n³) | O(n²) | Matrix chain multiplication |
| Bitmask DP | O(2^n × n²) | O(2^n × n) | TSP with n cities |
| Digit DP | O(digits × sum) | O(digits × sum) | With memoization |
| Probability DP | O(states × transitions) | O(states) | Depends on problem |
| Space Optimized | O(n) | O(1) | Rolling array technique |

## Applications

1. **Interval DP**: Matrix multiplication, palindrome problems, optimal BST
2. **Bitmask DP**: TSP, assignment problems, subset enumeration
3. **Digit DP**: Number theory problems, counting with constraints
4. **Probability DP**: Game theory, expected values, Markov chains
5. **Tree DP**: Tree traversal, path problems, subtree optimization

---

**Next**: [String Algorithms](string-algorithms.md/) | **Previous**: [Advanced DSA](README.md/) | **Up**: [Advanced DSA](README.md/)
