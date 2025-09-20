# Advanced Dynamic Programming

## Table of Contents
- [Introduction](#introduction)
- [Advanced DP Patterns](#advanced-dp-patterns)
- [Optimization Techniques](#optimization-techniques)
- [Complex State Management](#complex-state-management)
- [DP with Data Structures](#dp-with-data-structures)
- [Advanced Applications](#advanced-applications)

## Introduction

Advanced dynamic programming extends beyond basic DP concepts to handle complex optimization problems, state space management, and sophisticated algorithmic challenges.

## Advanced DP Patterns

### Digit DP

**Problem**: Count numbers in a range that satisfy certain digit constraints.

```go
// Count numbers with no consecutive 1s in binary
func countNumbersWithoutConsecutiveOnes(n int) int {
    // Convert to binary string
    s := fmt.Sprintf("%b", n)
    memo := make(map[string]int)
    
    var dp func(pos int, tight bool, prev int) int
    dp = func(pos int, tight bool, prev int) int {
        if pos == len(s) {
            return 1
        }
        
        key := fmt.Sprintf("%d-%t-%d", pos, tight, prev)
        if val, exists := memo[key]; exists {
            return val
        }
        
        limit := 9
        if tight {
            limit = int(s[pos] - '0')
        }
        
        result := 0
        for digit := 0; digit <= limit; digit++ {
            newTight := tight && (digit == limit)
            if prev == 1 && digit == 1 {
                continue // Skip consecutive 1s
            }
            result += dp(pos+1, newTight, digit)
        }
        
        memo[key] = result
        return result
    }
    
    return dp(0, true, 0)
}
```

### Probability DP

**Problem**: Calculate probabilities in stochastic processes.

```go
// Probability of reaching target in coin flipping
func coinFlipProbability(n int, target int) float64 {
    // dp[i][j] = probability of getting j heads in i flips
    dp := make([][]float64, n+1)
    for i := range dp {
        dp[i] = make([]float64, n+1)
    }
    
    dp[0][0] = 1.0 // Base case: 0 flips, 0 heads
    
    for i := 1; i <= n; i++ {
        for j := 0; j <= i; j++ {
            // Probability of getting j heads in i flips
            if j > 0 {
                dp[i][j] += dp[i-1][j-1] * 0.5 // Heads
            }
            dp[i][j] += dp[i-1][j] * 0.5 // Tails
        }
    }
    
    return dp[n][target]
}
```

### Game Theory DP

**Problem**: Optimal strategies in competitive games.

```go
// Minimax with alpha-beta pruning
func minimax(board [][]int, depth int, isMax bool, alpha, beta int) int {
    if depth == 0 || isGameOver(board) {
        return evaluateBoard(board)
    }
    
    if isMax {
        maxEval := math.MinInt32
        for _, move := range getPossibleMoves(board) {
            makeMove(board, move, true)
            eval := minimax(board, depth-1, false, alpha, beta)
            undoMove(board, move)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha {
                break // Alpha-beta pruning
            }
        }
        return maxEval
    } else {
        minEval := math.MaxInt32
        for _, move := range getPossibleMoves(board) {
            makeMove(board, move, false)
            eval := minimax(board, depth-1, true, alpha, beta)
            undoMove(board, move)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha {
                break // Alpha-beta pruning
            }
        }
        return minEval
    }
}
```

## Optimization Techniques

### Space Optimization

**Problem**: Reduce space complexity while maintaining correctness.

```go
// Space-optimized knapsack
func knapsackSpaceOptimized(weights, values []int, capacity int) int {
    n := len(weights)
    prev := make([]int, capacity+1)
    curr := make([]int, capacity+1)
    
    for i := 1; i <= n; i++ {
        for w := 0; w <= capacity; w++ {
            if weights[i-1] <= w {
                curr[w] = max(prev[w], values[i-1]+prev[w-weights[i-1]])
            } else {
                curr[w] = prev[w]
            }
        }
        prev, curr = curr, prev
    }
    
    return prev[capacity]
}

// Further space optimization - single array
func knapsackUltraOptimized(weights, values []int, capacity int) int {
    dp := make([]int, capacity+1)
    
    for i := 0; i < len(weights); i++ {
        // Iterate backwards to avoid using updated values
        for w := capacity; w >= weights[i]; w-- {
            dp[w] = max(dp[w], values[i]+dp[w-weights[i]])
        }
    }
    
    return dp[capacity]
}
```

### Time Optimization

**Problem**: Reduce time complexity through better state transitions.

```go
// Optimized longest common subsequence
func lcsOptimized(s1, s2 string) int {
    if len(s1) > len(s2) {
        s1, s2 = s2, s1
    }
    
    prev := make([]int, len(s1)+1)
    curr := make([]int, len(s1)+1)
    
    for i := 1; i <= len(s2); i++ {
        for j := 1; j <= len(s1); j++ {
            if s2[i-1] == s1[j-1] {
                curr[j] = 1 + prev[j-1]
            } else {
                curr[j] = max(prev[j], curr[j-1])
            }
        }
        prev, curr = curr, prev
    }
    
    return prev[len(s1)]
}
```

### Memory Optimization

**Problem**: Handle large state spaces efficiently.

```go
// Rolling hash for string matching
func rabinKarp(text, pattern string) []int {
    const base = 256
    const mod = 1000000007
    
    n, m := len(text), len(pattern)
    if m > n {
        return nil
    }
    
    // Calculate hash of pattern
    patternHash := 0
    for i := 0; i < m; i++ {
        patternHash = (patternHash*base + int(pattern[i])) % mod
    }
    
    // Calculate hash of first window
    windowHash := 0
    power := 1
    for i := 0; i < m; i++ {
        windowHash = (windowHash*base + int(text[i])) % mod
        if i < m-1 {
            power = (power * base) % mod
        }
    }
    
    var result []int
    
    // Slide the window
    for i := 0; i <= n-m; i++ {
        if windowHash == patternHash {
            if text[i:i+m] == pattern {
                result = append(result, i)
            }
        }
        
        if i < n-m {
            // Remove leading digit, add trailing digit
            windowHash = (windowHash - int(text[i])*power) % mod
            windowHash = (windowHash + mod) % mod
            windowHash = (windowHash*base + int(text[i+m])) % mod
        }
    }
    
    return result
}
```

## Complex State Management

### Multi-dimensional DP

**Problem**: Handle multiple constraints simultaneously.

```go
// 3D DP for longest common subsequence with k mismatches
func lcsWithMismatches(s1, s2 string, k int) int {
    n, m := len(s1), len(s2)
    dp := make([][][]int, n+1)
    for i := range dp {
        dp[i] = make([][]int, m+1)
        for j := range dp[i] {
            dp[i][j] = make([]int, k+1)
        }
    }
    
    for i := 1; i <= n; i++ {
        for j := 1; j <= m; j++ {
            for l := 0; l <= k; l++ {
                if s1[i-1] == s2[j-1] {
                    dp[i][j][l] = 1 + dp[i-1][j-1][l]
                } else {
                    dp[i][j][l] = max(dp[i-1][j][l], dp[i][j-1][l])
                    if l > 0 {
                        dp[i][j][l] = max(dp[i][j][l], 1+dp[i-1][j-1][l-1])
                    }
                }
            }
        }
    }
    
    return dp[n][m][k]
}
```

### State Compression

**Problem**: Represent complex states efficiently.

```go
// TSP with bitmask DP
func tspBitmask(graph [][]int) int {
    n := len(graph)
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, 1<<n)
        for j := range dp[i] {
            dp[i][j] = math.MaxInt32
        }
    }
    
    dp[0][1] = 0 // Start at city 0
    
    for mask := 1; mask < (1 << n); mask++ {
        for u := 0; u < n; u++ {
            if (mask & (1 << u)) == 0 {
                continue
            }
            
            for v := 0; v < n; v++ {
                if (mask & (1 << v)) == 0 && graph[u][v] > 0 {
                    newMask := mask | (1 << v)
                    dp[v][newMask] = min(dp[v][newMask], dp[u][mask]+graph[u][v])
                }
            }
        }
    }
    
    // Return to starting city
    result := math.MaxInt32
    for i := 1; i < n; i++ {
        if graph[i][0] > 0 {
            result = min(result, dp[i][(1<<n)-1]+graph[i][0])
        }
    }
    
    return result
}
```

### Interval DP

**Problem**: Optimize over intervals or ranges.

```go
// Matrix chain multiplication
func matrixChainMultiplication(dimensions []int) int {
    n := len(dimensions) - 1
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, n)
    }
    
    // Length of chain
    for l := 2; l <= n; l++ {
        for i := 0; i < n-l+1; i++ {
            j := i + l - 1
            dp[i][j] = math.MaxInt32
            
            for k := i; k < j; k++ {
                cost := dp[i][k] + dp[k+1][j] + 
                       dimensions[i]*dimensions[k+1]*dimensions[j+1]
                dp[i][j] = min(dp[i][j], cost)
            }
        }
    }
    
    return dp[0][n-1]
}

// Palindrome partitioning
func minPalindromePartitions(s string) int {
    n := len(s)
    dp := make([]int, n+1)
    
    // Precompute palindrome table
    isPalindrome := make([][]bool, n)
    for i := range isPalindrome {
        isPalindrome[i] = make([]bool, n)
    }
    
    // Single characters are palindromes
    for i := 0; i < n; i++ {
        isPalindrome[i][i] = true
    }
    
    // Check for palindromes of length 2
    for i := 0; i < n-1; i++ {
        if s[i] == s[i+1] {
            isPalindrome[i][i+1] = true
        }
    }
    
    // Check for palindromes of length 3 and more
    for length := 3; length <= n; length++ {
        for i := 0; i < n-length+1; i++ {
            j := i + length - 1
            if s[i] == s[j] && isPalindrome[i+1][j-1] {
                isPalindrome[i][j] = true
            }
        }
    }
    
    // DP for minimum cuts
    for i := 0; i < n; i++ {
        if isPalindrome[0][i] {
            dp[i] = 0
        } else {
            dp[i] = math.MaxInt32
            for j := 0; j < i; j++ {
                if isPalindrome[j+1][i] {
                    dp[i] = min(dp[i], dp[j]+1)
                }
            }
        }
    }
    
    return dp[n-1]
}
```

## DP with Data Structures

### Segment Tree DP

**Problem**: Range queries with DP optimization.

```go
// Range minimum query with DP
type SegmentTree struct {
    tree []int
    n    int
}

func NewSegmentTree(arr []int) *SegmentTree {
    n := len(arr)
    st := &SegmentTree{
        tree: make([]int, 4*n),
        n:    n,
    }
    st.build(arr, 0, 0, n-1)
    return st
}

func (st *SegmentTree) build(arr []int, node, start, end int) {
    if start == end {
        st.tree[node] = arr[start]
    } else {
        mid := (start + end) / 2
        st.build(arr, 2*node+1, start, mid)
        st.build(arr, 2*node+2, mid+1, end)
        st.tree[node] = min(st.tree[2*node+1], st.tree[2*node+2])
    }
}

func (st *SegmentTree) query(node, start, end, l, r int) int {
    if r < start || end < l {
        return math.MaxInt32
    }
    if l <= start && end <= r {
        return st.tree[node]
    }
    mid := (start + end) / 2
    return min(
        st.query(2*node+1, start, mid, l, r),
        st.query(2*node+2, mid+1, end, l, r),
    )
}
```

### Fenwick Tree DP

**Problem**: Prefix sum queries with DP.

```go
// Fenwick Tree for range sum queries
type FenwickTree struct {
    tree []int
    n    int
}

func NewFenwickTree(size int) *FenwickTree {
    return &FenwickTree{
        tree: make([]int, size+1),
        n:    size,
    }
}

func (ft *FenwickTree) update(index, delta int) {
    for index <= ft.n {
        ft.tree[index] += delta
        index += index & (-index)
    }
}

func (ft *FenwickTree) query(index int) int {
    sum := 0
    for index > 0 {
        sum += ft.tree[index]
        index -= index & (-index)
    }
    return sum
}

func (ft *FenwickTree) rangeQuery(l, r int) int {
    return ft.query(r) - ft.query(l-1)
}
```

## Advanced Applications

### Convex Hull Optimization

**Problem**: Optimize DP with convex hull properties.

```go
// Convex hull trick for DP optimization
type Line struct {
    m, b int // y = mx + b
}

type ConvexHullTrick struct {
    lines []Line
}

func (cht *ConvexHullTrick) addLine(m, b int) {
    line := Line{m, b}
    
    // Remove lines that are no longer optimal
    for len(cht.lines) >= 2 {
        n := len(cht.lines)
        l1, l2, l3 := cht.lines[n-2], cht.lines[n-1], line
        
        // Check if l2 is redundant
        if cht.intersection(l1, l3) <= cht.intersection(l1, l2) {
            cht.lines = cht.lines[:n-1]
        } else {
            break
        }
    }
    
    cht.lines = append(cht.lines, line)
}

func (cht *ConvexHullTrick) query(x int) int {
    if len(cht.lines) == 0 {
        return math.MaxInt32
    }
    
    // Binary search for optimal line
    left, right := 0, len(cht.lines)-1
    for left < right {
        mid := (left + right) / 2
        if cht.lines[mid].evaluate(x) < cht.lines[mid+1].evaluate(x) {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return cht.lines[left].evaluate(x)
}

func (l Line) evaluate(x int) int {
    return l.m*x + l.b
}

func (cht *ConvexHullTrick) intersection(l1, l2 Line) float64 {
    return float64(l2.b-l1.b) / float64(l1.m-l2.m)
}
```

### Divide and Conquer DP

**Problem**: Optimize DP with divide and conquer.

```go
// Divide and conquer optimization for DP
func divideAndConquerDP(n, k int, cost func(int, int) int) int {
    dp := make([][]int, k+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Base case
    for i := 0; i <= n; i++ {
        dp[0][i] = 0
    }
    
    for i := 1; i <= k; i++ {
        solve(i, 0, n-1, 0, n-1, dp, cost)
    }
    
    return dp[k][n]
}

func solve(i, l, r, optL, optR int, dp [][]int, cost func(int, int) int) {
    if l > r {
        return
    }
    
    mid := (l + r) / 2
    best := math.MaxInt32
    bestK := optL
    
    for k := optL; k <= min(mid, optR); k++ {
        current := dp[i-1][k] + cost(k+1, mid)
        if current < best {
            best = current
            bestK = k
        }
    }
    
    dp[i][mid] = best
    
    solve(i, l, mid-1, optL, bestK, dp, cost)
    solve(i, mid+1, r, bestK, optR, dp, cost)
}
```

### Monotonic Queue DP

**Problem**: Optimize DP with sliding window minimum.

```go
// Monotonic queue for sliding window minimum
type MonotonicQueue struct {
    deque []int
}

func (mq *MonotonicQueue) push(index int, arr []int) {
    for len(mq.deque) > 0 && arr[mq.deque[len(mq.deque)-1]] >= arr[index] {
        mq.deque = mq.deque[:len(mq.deque)-1]
    }
    mq.deque = append(mq.deque, index)
}

func (mq *MonotonicQueue) pop(index int) {
    if len(mq.deque) > 0 && mq.deque[0] == index {
        mq.deque = mq.deque[1:]
    }
}

func (mq *MonotonicQueue) getMin(arr []int) int {
    if len(mq.deque) == 0 {
        return math.MaxInt32
    }
    return arr[mq.deque[0]]
}

// DP with monotonic queue optimization
func dpWithMonotonicQueue(arr []int, k int) int {
    n := len(arr)
    dp := make([]int, n)
    mq := &MonotonicQueue{}
    
    for i := 0; i < n; i++ {
        // Remove elements outside window
        mq.pop(i - k)
        
        // Add current element
        mq.push(i, arr)
        
        // Update DP
        if i < k {
            dp[i] = arr[i]
        } else {
            dp[i] = mq.getMin(arr) + arr[i]
        }
    }
    
    return dp[n-1]
}
```

## Conclusion

Advanced dynamic programming techniques provide:

1. **Efficiency**: Optimized algorithms for complex problems
2. **Scalability**: Handling large state spaces effectively
3. **Flexibility**: Adapting to various problem constraints
4. **Optimization**: Space and time complexity improvements
5. **Patterns**: Reusable solutions for common problem types
6. **Data Structures**: Integration with advanced data structures
7. **Mathematical**: Leveraging mathematical properties for optimization

Mastering these techniques prepares you for complex algorithmic challenges in technical interviews and real-world applications.

## Additional Resources

- [Advanced DP Patterns](https://www.advanceddppatterns.com/)
- [Optimization Techniques](https://www.optimizationtechniques.com/)
- [State Management](https://www.statemanagement.com/)
- [Data Structures in DP](https://www.datastructuresindp.com/)
- [Mathematical Optimization](https://www.mathematicaloptimization.com/)
- [Algorithm Design](https://www.algorithmdesign.com/)
- [Competitive Programming](https://www.competitiveprogramming.com/)
- [Dynamic Programming Guide](https://www.dynamicprogrammingguide.com/)
