# Dynamic Programming Pattern

> **Master dynamic programming techniques with Go implementations**

## 📋 Problems

### **1D DP**

- [Climbing Stairs](ClimbingStairs.md) - Fibonacci sequence variation
- [House Robber](HouseRobber.md) - Maximum sum with constraints
- [Longest Increasing Subsequence](LongestIncreasingSubsequence.md) - LIS with O(n log n) solution
- [Word Break](WordBreak.md) - String segmentation problem
- [Decode Ways](../Strings/DecodeWays.md) - String to number conversion

### **2D DP**

- [Unique Paths](UniquePaths.md) - Grid traversal counting
- [Minimum Path Sum](MinimumPathSum.md) - Grid optimization
- [Longest Common Subsequence](LongestCommonSubsequence.md) - String comparison
- [Edit Distance](EditDistance.md) - String transformation
- [Knapsack](Knapsack.md) - Classic optimization problem

### **Advanced DP**

- [Coin Change](CoinChange.md) - Minimum coins problem
- [Palindrome Partitioning](../Backtracking/PalindromePartitioning.md) - String partitioning
- [Regular Expression Matching](../Strings/RegularExpressionMatching.md) - Pattern matching
- [Wildcard Matching](../Strings/WildcardMatching.md) - Advanced pattern matching
- [Maximum Product Subarray](../Arrays/MaximumProductSubarray.md) - Array optimization

---

## 🎯 Key Concepts

### **Dynamic Programming Principles**

**Detailed Explanation:**
Dynamic Programming (DP) is a powerful algorithmic technique that solves complex problems by breaking them down into simpler subproblems and storing the results to avoid redundant calculations. It's particularly effective for optimization problems where the same subproblems are solved multiple times.

**Core Principles:**

1. **Optimal Substructure**: The optimal solution to a problem contains optimal solutions to its subproblems. This means we can build the solution by combining optimal solutions to smaller problems.

2. **Overlapping Subproblems**: The same subproblems are solved multiple times in a recursive approach. DP avoids this redundancy by storing results.

3. **Memoization**: Store the results of subproblems in a data structure (usually a hash map or array) to avoid recomputation.

4. **Tabulation**: Build the solution bottom-up by filling a table with subproblem results, starting from the base cases.

**Why DP Works:**

- **Efficiency**: Transforms exponential time complexity to polynomial time
- **Reusability**: Once a subproblem is solved, its result can be reused
- **Systematic Approach**: Provides a structured way to solve complex problems
- **Space-Time Trade-off**: Uses additional space to save computation time

**Mathematical Foundation:**

```
T(n) = T(n-1) + T(n-2) + O(1)  // Without DP: O(2^n)
T(n) = T(n-1) + O(1)           // With DP: O(n)
```

### **DP Patterns**

**Detailed Explanation:**
Different types of problems require different DP patterns. Understanding these patterns helps in quickly identifying the right approach for a given problem.

**1D DP (Linear Problems):**

- **Use Case**: Problems where the state depends on a single dimension
- **Examples**: Fibonacci, House Robber, Climbing Stairs
- **State Definition**: `dp[i]` represents the optimal solution for the first `i` elements
- **Transition**: `dp[i] = f(dp[i-1], dp[i-2], ...)`

**2D DP (Grid/Matrix Problems):**

- **Use Case**: Problems involving two dimensions (grids, strings, matrices)
- **Examples**: Unique Paths, LCS, Edit Distance
- **State Definition**: `dp[i][j]` represents the optimal solution for subproblem involving first `i` and `j` elements
- **Transition**: `dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])`

**Interval DP:**

- **Use Case**: Problems on ranges or intervals
- **Examples**: Matrix Chain Multiplication, Palindrome Partitioning
- **State Definition**: `dp[i][j]` represents the optimal solution for interval from `i` to `j`
- **Transition**: `dp[i][j] = min/max(dp[i][k] + dp[k+1][j] + cost)`

**Tree DP:**

- **Use Case**: Problems on tree structures
- **Examples**: Binary Tree Cameras, House Robber III
- **State Definition**: `dp[node]` represents the optimal solution for subtree rooted at `node`
- **Transition**: Combine results from child nodes

**State Machine DP:**

- **Use Case**: Problems with multiple states or transitions
- **Examples**: Stock Trading, Buy and Sell Stock
- **State Definition**: `dp[i][state]` represents the optimal solution at position `i` in state `state`
- **Transition**: `dp[i][state] = f(dp[i-1][prev_state], transition_cost)`

### **Implementation Approaches**

**Detailed Explanation:**
There are two main approaches to implement dynamic programming solutions, each with its own advantages and use cases.

**Top-Down (Memoization):**

- **Approach**: Start with the main problem and recursively solve subproblems
- **Implementation**: Use recursion with memoization
- **Advantages**: More intuitive, follows natural problem structure
- **Disadvantages**: Recursion overhead, potential stack overflow
- **Best For**: When the problem structure is naturally recursive

**Bottom-Up (Tabulation):**

- **Approach**: Start with base cases and build up to the main problem
- **Implementation**: Use iterative loops with tables
- **Advantages**: No recursion overhead, better space control
- **Disadvantages**: Less intuitive, requires careful ordering
- **Best For**: When you need precise control over computation order

**Space Optimization:**

- **Technique**: Reduce space complexity by reusing space
- **Methods**: Use rolling arrays, swap variables, or compute on-the-fly
- **Trade-off**: Slightly more complex code for better space efficiency
- **Best For**: When space is a constraint or for very large inputs

**Discussion Questions & Answers:**

**Q1: How do you identify when a problem can be solved using dynamic programming?**

**Answer:** Look for these characteristics:

- **Optimization Problem**: Finding minimum, maximum, or count
- **Decision Problem**: Yes/No questions with constraints
- **Overlapping Subproblems**: Same subproblems appear multiple times
- **Optimal Substructure**: Optimal solution contains optimal subproblems
- **Recursive Structure**: Problem can be broken down into smaller versions
- **Memoization Opportunity**: Storing results would improve efficiency

**Q2: What are the trade-offs between top-down and bottom-up approaches?**

**Answer:** Trade-offs include:

- **Intuition**: Top-down is more intuitive, bottom-up requires careful planning
- **Performance**: Bottom-up is generally faster due to no recursion overhead
- **Space**: Bottom-up allows better space optimization
- **Debugging**: Top-down is easier to debug due to natural recursion
- **Implementation**: Top-down is often easier to implement initially
- **Stack Overflow**: Top-down can cause stack overflow for deep recursion

**Q3: How do you optimize space complexity in dynamic programming solutions?**

**Answer:** Space optimization strategies:

- **Rolling Arrays**: Use only the necessary previous states
- **Variable Swapping**: Swap current and previous arrays
- **In-place Updates**: Update the same array when possible
- **State Compression**: Use bit manipulation for boolean states
- **Lazy Evaluation**: Compute values only when needed
- **Memory Pooling**: Reuse allocated memory for multiple problems

---

## 🛠️ Go-Specific Tips

### **Memoization with Maps**

**Detailed Explanation:**
Go's map data structure is perfect for memoization in dynamic programming. Maps provide O(1) average case lookup and insertion, making them ideal for storing subproblem results. However, there are important considerations for thread safety and memory management.

**Implementation Patterns:**

```go
// Global memoization map
var memo = make(map[int]int)

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }

    if val, exists := memo[n]; exists {
        return val
    }

    memo[n] = fibonacci(n-1) + fibonacci(n-2)
    return memo[n]
}
```

**Thread-Safe Memoization:**

```go
type SafeMemo struct {
    mu   sync.RWMutex
    data map[int]int
}

func (m *SafeMemo) Get(key int) (int, bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    val, exists := m.data[key]
    return val, exists
}

func (m *SafeMemo) Set(key, value int) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
}
```

**Discussion Questions & Answers:**

**Q1: When should you use global vs local memoization maps?**

**Answer:** Choose based on context:

- **Global Maps**: When the same subproblems are shared across multiple function calls
- **Local Maps**: When each function call should have its own memoization space
- **Thread Safety**: Use local maps or thread-safe global maps for concurrent access
- **Memory Management**: Local maps are automatically garbage collected
- **Performance**: Global maps avoid repeated allocation overhead

**Q2: How do you handle memory management with large memoization maps?**

**Answer:** Memory management strategies:

- **Size Limits**: Set maximum size for memoization maps
- **LRU Eviction**: Remove least recently used entries when limit reached
- **Periodic Cleanup**: Clear maps periodically to prevent memory leaks
- **Weak References**: Use weak references for large objects
- **Memory Pooling**: Reuse map instances for similar problems

### **2D DP Table**

**Detailed Explanation:**
2D dynamic programming tables are essential for problems involving two dimensions. Go's slice of slices provides an efficient way to implement these tables, but proper initialization and memory management are crucial.

**Implementation Patterns:**

```go
// Create 2D slice for DP table
dp := make([][]int, m)
for i := range dp {
    dp[i] = make([]int, n)
}

// Initialize base cases
dp[0][0] = 1

// Fill the table
for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        // DP transition
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
    }
}
```

**Memory-Efficient Initialization:**

```go
// Pre-allocate all memory at once
dp := make([][]int, m)
data := make([]int, m*n)
for i := range dp {
    dp[i] = data[i*n : (i+1)*n]
}
```

**Discussion Questions & Answers:**

**Q1: What are the performance implications of different 2D slice initialization methods?**

**Answer:** Performance characteristics:

- **Separate Allocation**: `make([]int, n)` for each row - slower but more flexible
- **Single Allocation**: Pre-allocate all memory - faster but less flexible
- **Memory Layout**: Single allocation provides better cache locality
- **Garbage Collection**: Single allocation reduces GC pressure
- **Access Patterns**: Consider how you'll access the data for optimization

**Q2: How do you optimize 2D DP table access patterns in Go?**

**Answer:** Optimization strategies:

- **Row-Major Order**: Access elements row by row for better cache locality
- **Column-Major Order**: Access elements column by column if needed
- **Block Processing**: Process data in blocks to improve cache utilization
- **Memory Alignment**: Ensure proper memory alignment for better performance
- **Prefetching**: Use prefetch hints for predictable access patterns

### **Space Optimization**

**Detailed Explanation:**
Space optimization is crucial for dynamic programming problems with large inputs. Go's slice manipulation and variable swapping provide elegant ways to reduce space complexity from O(n²) to O(n) or even O(1).

**Implementation Patterns:**

```go
// Instead of 2D table, use 1D array
prev := make([]int, n)
curr := make([]int, n)

for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        curr[j] = prev[j] + curr[j-1]
    }
    prev, curr = curr, prev
}
```

**Advanced Space Optimization:**

```go
// In-place updates when possible
func uniquePaths(m, n int) int {
    dp := make([]int, n)
    for i := range dp {
        dp[i] = 1
    }

    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            dp[j] += dp[j-1]
        }
    }

    return dp[n-1]
}
```

**Discussion Questions & Answers:**

**Q1: When is space optimization worth the added complexity?**

**Answer:** Space optimization is worth it when:

- **Memory Constraints**: Limited memory available
- **Large Inputs**: Input size approaches memory limits
- **Performance Critical**: Memory allocation is a bottleneck
- **Scalability**: Need to handle larger inputs in the future
- **Cost**: Memory usage has significant cost implications
- **Cache Efficiency**: Reduced memory usage improves cache performance

**Q2: How do you debug space-optimized DP solutions?**

**Answer:** Debugging strategies:

- **Add Logging**: Log intermediate values to understand the algorithm
- **Visualization**: Create visual representations of the DP table
- **Step-by-Step**: Implement both optimized and unoptimized versions
- **Unit Tests**: Test with small inputs where you can verify manually
- **Memory Profiling**: Use Go's memory profiler to understand memory usage
- **Incremental Development**: Start with unoptimized version, then optimize

### **Common DP Patterns in Go**

**Detailed Explanation:**
These patterns represent the most common dynamic programming scenarios you'll encounter. Understanding these patterns helps in quickly implementing solutions for similar problems.

**Implementation Patterns:**

```go
// 1. Fibonacci pattern
func dp(n int) int {
    if n <= 1 {
        return n
    }

    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    return b
}

// 2. LIS pattern
func lis(nums []int) int {
    dp := make([]int, len(nums))
    for i := range dp {
        dp[i] = 1
    }

    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
    }

    return max(dp...)
}
```

**Advanced Patterns:**

```go
// 3. Knapsack pattern
func knapsack(weights, values []int, capacity int) int {
    dp := make([]int, capacity+1)

    for i := 0; i < len(weights); i++ {
        for w := capacity; w >= weights[i]; w-- {
            dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
        }
    }

    return dp[capacity]
}

// 4. Edit distance pattern
func editDistance(s1, s2 string) int {
    m, n := len(s1), len(s2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }

    for i := 0; i <= m; i++ {
        for j := 0; j <= n; j++ {
            if i == 0 {
                dp[i][j] = j
            } else if j == 0 {
                dp[i][j] = i
            } else if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
    }

    return dp[m][n]
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose the right DP pattern for a given problem?**

**Answer:** Pattern selection criteria:

- **Problem Type**: Optimization, counting, or decision problem
- **Input Structure**: Linear, 2D, tree, or graph
- **State Dependencies**: How current state depends on previous states
- **Constraints**: Time and space complexity requirements
- **Problem Size**: Expected input size and memory constraints
- **Similar Problems**: Look for problems with similar structure

**Q2: What are the common pitfalls when implementing DP patterns in Go?**

**Answer:** Common pitfalls include:

- **Index Errors**: Off-by-one errors in array indexing
- **Initialization**: Incorrect base case initialization
- **State Transitions**: Wrong recurrence relations
- **Memory Management**: Not properly managing slice memory
- **Concurrency**: Race conditions in concurrent access
- **Type Safety**: Incorrect type conversions and assertions

---

## 🎯 Interview Tips

### **How to Identify DP Problems**

1. **Optimization Problem**: Find minimum/maximum/count
2. **Decision Problem**: Yes/No with constraints
3. **Overlapping Subproblems**: Same subproblems appear multiple times
4. **Optimal Substructure**: Optimal solution contains optimal subproblems

### **DP Problem Solving Steps**

1. **Identify the problem type** (1D, 2D, interval, etc.)
2. **Define the state** (what does dp[i] represent?)
3. **Find the recurrence relation** (how to compute dp[i] from previous states?)
4. **Identify base cases** (initial values)
5. **Choose implementation approach** (top-down or bottom-up)
6. **Optimize space** (if possible)

### **Common Mistakes to Avoid**

- **Incorrect state definition**: Make sure dp[i] represents what you think it does
- **Wrong base cases**: Initialize correctly
- **Off-by-one errors**: Be careful with array indices
- **Memory issues**: Use space optimization for large inputs
