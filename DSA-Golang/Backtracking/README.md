# Backtracking Pattern

> **Master backtracking algorithms and recursive problem solving with Go implementations**

## 📋 Problems

### **Permutations and Combinations**
- [Permutations](./Permutations.md) - Generate all permutations
- [Permutations II](./PermutationsII.md) - Permutations with duplicates
- [Combinations](./Combinations.md) - Generate all combinations
- [Combination Sum](./CombinationSum.md) - Find combinations that sum to target
- [Combination Sum II](./CombinationSumII.md) - Combination sum with duplicates

### **N-Queens and Similar Problems**
- [N-Queens](./NQueens.md) - Place N queens on N×N board
- [N-Queens II](./NQueensII.md) - Count number of solutions
- [Sudoku Solver](./SudokuSolver.md) - Solve Sudoku puzzle
- [Word Search](./WordSearch.md) - Find word in 2D grid
- [Word Search II](./WordSearchII.md) - Find multiple words

### **Subset Problems**
- [Subsets](./Subsets.md) - Generate all subsets
- [Subsets II](./SubsetsII.md) - Subsets with duplicates
- [Generate Parentheses](./GenerateParentheses.md) - Generate valid parentheses
- [Letter Combinations of a Phone Number](./LetterCombinations.md) - Phone number combinations
- [Restore IP Addresses](./RestoreIPAddresses.md) - Valid IP address restoration

### **Advanced Backtracking**
- [Palindrome Partitioning](./PalindromePartitioning.md) - Partition string into palindromes
- [Word Break II](./WordBreakII.md) - All possible word breaks
- [Expression Add Operators](./ExpressionAddOperators.md) - Add operators to expression
- [Remove Invalid Parentheses](./RemoveInvalidParentheses.md) - Remove minimum parentheses
- [Word Pattern II](./WordPatternII.md) - Match string to pattern

---

## 🎯 Key Concepts

### **Backtracking Algorithm Structure**
1. **Choose**: Make a choice at each step
2. **Explore**: Recursively explore the choice
3. **Unchoose**: Undo the choice (backtrack)
4. **Base Case**: Stop when solution is found or no more choices

### **When to Use Backtracking**
- **Generate All Solutions**: Find all possible solutions
- **Constraint Satisfaction**: Find solutions that satisfy constraints
- **Combinatorial Problems**: Permutations, combinations, subsets
- **Search Problems**: Find paths, configurations, arrangements

### **Common Backtracking Patterns**
- **State Space Tree**: Represent all possible states
- **Pruning**: Skip branches that can't lead to solutions
- **Memoization**: Cache results to avoid recomputation
- **Iterative Backtracking**: Use stack instead of recursion

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
