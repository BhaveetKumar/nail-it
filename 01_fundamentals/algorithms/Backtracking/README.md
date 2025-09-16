# Backtracking Pattern

> **Master backtracking algorithms and recursive problem solving with Go implementations**

## 📋 Problems

### **Permutations and Combinations**

- [Permutations](Permutations.md/) - Generate all permutations
- [Permutations II](PermutationsII.md/) - Permutations with duplicates
- [Combinations](Combinations.md/) - Generate all combinations
- [Combination Sum](CombinationSum.md/) - Find combinations that sum to target
- [Combination Sum II](CombinationSumII.md/) - Combination sum with duplicates

### **N-Queens and Similar Problems**

- [N-Queens](NQueens.md/) - Place N queens on N×N board
- [N-Queens II](NQueensII.md/) - Count number of solutions
- [Sudoku Solver](SudokuSolver.md/) - Solve Sudoku puzzle
- [Word Search](WordSearch.md/) - Find word in 2D grid
- [Word Search II](WordSearchII.md/) - Find multiple words

### **Subset Problems**

- [Subsets](Subsets.md/) - Generate all subsets
- [Subsets II](SubsetsII.md/) - Subsets with duplicates
- [Generate Parentheses](GenerateParentheses.md/) - Generate valid parentheses
- [Letter Combinations of a Phone Number](LetterCombinations.md/) - Phone number combinations
- [Restore IP Addresses](RestoreIPAddresses.md/) - Valid IP address restoration

### **Advanced Backtracking**

- [Palindrome Partitioning](PalindromePartitioning.md/) - Partition string into palindromes
- [Word Break II](WordBreakII.md/) - All possible word breaks
- [Expression Add Operators](ExpressionAddOperators.md/) - Add operators to expression
- [Remove Invalid Parentheses](RemoveInvalidParentheses.md/) - Remove minimum parentheses
- [Word Pattern II](WordPatternII.md/) - Match string to pattern

---

## 🎯 Key Concepts

### **Backtracking Algorithm Structure**

**Detailed Explanation:**
Backtracking is a systematic method for solving problems by exploring all possible solutions through recursive exploration and undoing choices when they don't lead to valid solutions. It's particularly effective for constraint satisfaction problems and combinatorial optimization.

**Core Algorithm Steps:**

1. **Choose**: Make a choice at each step from available options
2. **Explore**: Recursively explore the consequences of that choice
3. **Unchoose**: Undo the choice (backtrack) if it doesn't lead to a solution
4. **Base Case**: Stop when a solution is found or no more choices are available

**Why Backtracking Works:**

- **Systematic Exploration**: Ensures all possible solutions are considered
- **Constraint Satisfaction**: Only explores paths that satisfy given constraints
- **Memory Efficient**: Uses recursion stack instead of storing all states
- **Flexible**: Can be adapted to various problem types

**Mathematical Foundation:**

```
Time Complexity: O(b^d) where b is branching factor, d is depth
Space Complexity: O(d) for recursion stack
```

### **When to Use Backtracking**

**Detailed Explanation:**
Backtracking is most effective for problems that can be modeled as a search through a state space tree where you need to find all valid solutions or the best solution among many possibilities.

**Problem Characteristics:**

- **Generate All Solutions**: Find all possible arrangements, combinations, or configurations
- **Constraint Satisfaction**: Find solutions that satisfy specific constraints
- **Combinatorial Problems**: Permutations, combinations, subsets, arrangements
- **Search Problems**: Find paths, configurations, or valid arrangements
- **Decision Problems**: Yes/No questions with multiple valid answers

**When NOT to Use Backtracking:**

- **Single Solution**: When you only need one solution (use other search algorithms)
- **Large Search Space**: When the search space is too large (use heuristics)
- **Real-time Requirements**: When you need immediate results
- **Memory Constraints**: When recursion depth is too deep

**Alternative Approaches:**

- **Dynamic Programming**: For optimization problems with overlapping subproblems
- **Greedy Algorithms**: For problems with optimal substructure
- **BFS/DFS**: For pathfinding without constraints
- **Heuristic Search**: For large search spaces

### **Common Backtracking Patterns**

**Detailed Explanation:**
Understanding common patterns helps in quickly identifying and implementing backtracking solutions for various problem types.

**State Space Tree:**

- **Definition**: A tree representing all possible states of the problem
- **Nodes**: Represent partial solutions or states
- **Edges**: Represent choices or transitions between states
- **Leaves**: Represent complete solutions or dead ends
- **Implementation**: Implicit tree through recursive calls

**Pruning:**

- **Purpose**: Skip branches that can't lead to valid solutions
- **Types**: Constraint-based pruning, bound-based pruning, symmetry breaking
- **Benefits**: Significantly reduces search space and improves performance
- **Implementation**: Early termination when constraints are violated

**Memoization:**

- **Purpose**: Cache results to avoid recomputing the same subproblems
- **Use Case**: When the same state can be reached through different paths
- **Implementation**: Use maps or arrays to store computed results
- **Trade-off**: Space vs time complexity

**Iterative Backtracking:**

- **Purpose**: Use explicit stack instead of recursion
- **Benefits**: Avoids stack overflow, better control over execution
- **Implementation**: Manual stack management with state objects
- **Use Case**: Deep recursion or when you need to pause/resume execution

**Discussion Questions & Answers:**

**Q1: How do you optimize backtracking algorithms for better performance?**

**Answer:** Optimization strategies:

- **Pruning**: Implement constraint checking early to skip invalid branches
- **Memoization**: Cache results for repeated subproblems
- **Ordering**: Try most promising choices first to find solutions faster
- **Symmetry Breaking**: Avoid exploring symmetric solutions
- **Bounds**: Use upper/lower bounds to prune branches
- **Data Structures**: Use efficient data structures for state representation
- **Iterative Implementation**: Use explicit stack for deep recursion

**Q2: What are the common pitfalls when implementing backtracking in Go?**

**Answer:** Common pitfalls include:

- **Slice Sharing**: Not creating copies of slices when needed, leading to shared state
- **State Management**: Not properly restoring state after backtracking
- **Memory Leaks**: Not cleaning up resources in recursive calls
- **Infinite Recursion**: Not implementing proper base cases
- **Constraint Checking**: Not validating constraints at the right time
- **Performance**: Not implementing pruning or memoization when beneficial
- **Concurrency**: Not handling concurrent access to shared state

**Q3: How do you handle backtracking problems with complex constraints?**

**Answer:** Complex constraint handling:

- **Constraint Propagation**: Use constraint propagation to reduce search space
- **Forward Checking**: Check constraints before making choices
- **Arc Consistency**: Ensure all constraints are satisfied
- **Constraint Satisfaction**: Use CSP (Constraint Satisfaction Problem) techniques
- **Heuristic Ordering**: Order choices based on constraint satisfaction likelihood
- **Constraint Relaxation**: Temporarily relax constraints to find partial solutions
- **Multi-level Backtracking**: Handle constraints at different levels of the search

---

## 🛠️ Go-Specific Tips

### **Basic Backtracking Template**

```go
func backtrack(path []int, choices []int) {
    // Base case: found a solution
    if isComplete(path) {
        result = append(result, append([]int{}, path...))
        return
    }

    // Try each choice
    for _, choice := range choices {
        // Make choice
        path = append(path, choice)

        // Explore
        backtrack(path, getNextChoices(choice))

        // Unchoose (backtrack)
        path = path[:len(path)-1]
    }
}
```

### **N-Queens Implementation**

```go
func solveNQueens(n int) [][]string {
    var result [][]string
    board := make([]int, n) // board[i] = column of queen in row i

    var backtrack func(int)
    backtrack = func(row int) {
        if row == n {
            // Found a solution
            solution := make([]string, n)
            for i := 0; i < n; i++ {
                rowStr := make([]byte, n)
                for j := 0; j < n; j++ {
                    if j == board[i] {
                        rowStr[j] = 'Q'
                    } else {
                        rowStr[j] = '.'
                    }
                }
                solution[i] = string(rowStr)
            }
            result = append(result, solution)
            return
        }

        // Try placing queen in each column
        for col := 0; col < n; col++ {
            if isValid(board, row, col) {
                board[row] = col
                backtrack(row + 1)
                // No need to undo board[row] as it gets overwritten
            }
        }
    }

    backtrack(0)
    return result
}

func isValid(board []int, row, col int) bool {
    for i := 0; i < row; i++ {
        // Check column
        if board[i] == col {
            return false
        }
        // Check diagonals
        if abs(board[i]-col) == abs(i-row) {
            return false
        }
    }
    return true
}
```

### **Permutations with Backtracking**

```go
func permute(nums []int) [][]int {
    var result [][]int
    var path []int
    used := make([]bool, len(nums))

    var backtrack func()
    backtrack = func() {
        if len(path) == len(nums) {
            result = append(result, append([]int{}, path...))
            return
        }

        for i := 0; i < len(nums); i++ {
            if !used[i] {
                // Choose
                path = append(path, nums[i])
                used[i] = true

                // Explore
                backtrack()

                // Unchoose
                path = path[:len(path)-1]
                used[i] = false
            }
        }
    }

    backtrack()
    return result
}
```

### **Word Search with Backtracking**

```go
func exist(board [][]byte, word string) bool {
    if len(board) == 0 || len(board[0]) == 0 {
        return false
    }

    m, n := len(board), len(board[0])
    visited := make([][]bool, m)
    for i := range visited {
        visited[i] = make([]bool, n)
    }

    var backtrack func(int, int, int) bool
    backtrack = func(row, col, index int) bool {
        if index == len(word) {
            return true
        }

        if row < 0 || row >= m || col < 0 || col >= n ||
           visited[row][col] || board[row][col] != word[index] {
            return false
        }

        // Mark as visited
        visited[row][col] = true

        // Explore all four directions
        directions := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
        for _, dir := range directions {
            if backtrack(row+dir[0], col+dir[1], index+1) {
                return true
            }
        }

        // Backtrack
        visited[row][col] = false
        return false
    }

    // Try starting from each cell
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if backtrack(i, j, 0) {
                return true
            }
        }
    }

    return false
}
```

---

## 🎯 Interview Tips

### **How to Identify Backtracking Problems**

1. **Generate All Solutions**: Find all possible arrangements
2. **Constraint Satisfaction**: Find solutions that meet criteria
3. **Decision Tree**: Problem can be represented as a tree
4. **Undo Operations**: Need to backtrack and try alternatives

### **Common Backtracking Problem Patterns**

- **Permutations**: Arrange elements in different orders
- **Combinations**: Select subsets of elements
- **N-Queens**: Place objects with constraints
- **Word Search**: Find paths in 2D grid
- **Sudoku**: Fill grid with constraints

### **Optimization Tips**

- **Pruning**: Skip branches that can't lead to solutions
- **Memoization**: Cache results to avoid recomputation
- **Early Termination**: Stop when solution is found
- **State Compression**: Use bit manipulation for state representation
