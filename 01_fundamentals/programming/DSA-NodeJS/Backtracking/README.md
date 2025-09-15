# Backtracking Problems

Backtracking is a systematic way to iterate through all possible configurations of a search space. It's particularly useful for problems involving:

- **Constraint Satisfaction**: Finding solutions that satisfy given constraints
- **Combinatorial Problems**: Generating all possible combinations/permutations
- **Decision Problems**: Finding paths or sequences that meet criteria
- **Optimization**: Finding the best solution among all possibilities

## Key Concepts

### Backtracking Template
```javascript
function backtrack(path, choices) {
    // Base case: solution found
    if (isComplete(path)) {
        result.push([...path]);
        return;
    }
    
    // Try each choice
    for (let choice of choices) {
        // Make choice
        path.push(choice);
        
        // Recurse
        if (isValid(path)) {
            backtrack(path, getNextChoices(choice));
        }
        
        // Backtrack (undo choice)
        path.pop();
    }
}
```

### Common Patterns
1. **Generate All Combinations**: Find all possible combinations of elements
2. **Generate All Permutations**: Find all possible arrangements
3. **Path Finding**: Find paths in graphs/trees
4. **Constraint Satisfaction**: Solve problems with constraints
5. **Optimization**: Find optimal solutions

## Problems

### 1. [Generate Parentheses](./GenerateParentheses.md)
Generate all valid combinations of n pairs of parentheses.

### 2. [N-Queens](./NQueens.md)
Place n queens on an n×n chessboard so no two queens attack each other.

### 3. [Combination Sum](./CombinationSum.md)
Find all unique combinations that sum to a target value.

### 4. [Permutations](./Permutations.md)
Generate all possible permutations of an array.

### 5. [Subsets](./Subsets.md)
Generate all possible subsets of an array.

### 6. [Word Search](./WordSearch.md)
Find if a word exists in a 2D board using adjacent cells.

### 7. [Sudoku Solver](./SudokuSolver.md)
Solve a 9×9 Sudoku puzzle using backtracking.

### 8. [Letter Combinations](./LetterCombinations.md)
Generate all possible letter combinations from phone number digits.

### 9. [Palindrome Partitioning](./PalindromePartitioning.md)
Partition a string into palindromic substrings.

### 10. [Restore IP Addresses](./RestoreIPAddresses.md)
Restore valid IP addresses from a string of digits.

## Time & Space Complexity

| Problem | Time Complexity | Space Complexity |
|---------|----------------|------------------|
| Generate Parentheses | O(4^n / √n) | O(n) |
| N-Queens | O(n!) | O(n²) |
| Combination Sum | O(2^n) | O(target) |
| Permutations | O(n!) | O(n) |
| Subsets | O(2^n) | O(n) |
| Word Search | O(m×n×4^L) | O(L) |
| Sudoku Solver | O(9^(n×n)) | O(n²) |

Where:
- n = input size
- m×n = board dimensions
- L = word length
- target = target sum

## Tips for Backtracking Problems

1. **Identify the State Space**: What are all possible states?
2. **Define Constraints**: What makes a state valid?
3. **Choose Recursion Parameters**: What information to pass down?
4. **Implement Pruning**: Skip invalid branches early
5. **Handle Base Cases**: When to stop recursion
6. **Backtrack Properly**: Undo changes when returning

## Common Mistakes

1. **Forgetting to Backtrack**: Not undoing changes
2. **Incorrect Base Case**: Wrong termination condition
3. **Inefficient Pruning**: Not skipping invalid branches early
4. **State Management**: Not properly managing the current state
5. **Duplicate Solutions**: Not handling duplicates correctly
