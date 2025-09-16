# N-Queens

## Problem Statement

The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

**Example 1:**
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above.
```

**Example 2:**
```
Input: n = 1
Output: [["Q"]]
```

## Approach

### Backtracking Approach
1. Place queens row by row
2. For each row, try placing a queen in each column
3. Check if the placement is valid (no conflicts)
4. If valid, recurse to next row
5. If invalid or no solution found, backtrack

**Time Complexity:** O(n!) - In the worst case, we try all permutations
**Space Complexity:** O(n²) - For the board and recursion stack

## Solution

```javascript
/**
 * Solve the n-queens puzzle
 * @param {number} n - Size of the chessboard
 * @return {string[][]} - All distinct solutions
 */
function solveNQueens(n) {
    const result = [];
    const board = Array(n).fill().map(() => Array(n).fill('.'));
    
    function isValid(row, col) {
        // Check column
        for (let i = 0; i < row; i++) {
            if (board[i][col] === 'Q') {
                return false;
            }
        }
        
        // Check diagonal (top-left to bottom-right)
        for (let i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] === 'Q') {
                return false;
            }
        }
        
        // Check diagonal (top-right to bottom-left)
        for (let i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] === 'Q') {
                return false;
            }
        }
        
        return true;
    }
    
    function backtrack(row) {
        // Base case: all queens placed
        if (row === n) {
            result.push(board.map(row => row.join('')));
            return;
        }
        
        // Try placing queen in each column of current row
        for (let col = 0; col < n; col++) {
            if (isValid(row, col)) {
                board[row][col] = 'Q';
                backtrack(row + 1);
                board[row][col] = '.'; // Backtrack
            }
        }
    }
    
    backtrack(0);
    return result;
}

// Optimized version with sets for faster conflict checking
function solveNQueensOptimized(n) {
    const result = [];
    const board = Array(n).fill().map(() => Array(n).fill('.'));
    
    const cols = new Set();
    const diag1 = new Set(); // diagonal from top-left to bottom-right
    const diag2 = new Set(); // diagonal from top-right to bottom-left
    
    function backtrack(row) {
        if (row === n) {
            result.push(board.map(row => row.join('')));
            return;
        }
        
        for (let col = 0; col < n; col++) {
            const d1 = row - col; // diagonal 1
            const d2 = row + col; // diagonal 2
            
            if (cols.has(col) || diag1.has(d1) || diag2.has(d2)) {
                continue;
            }
            
            // Place queen
            board[row][col] = 'Q';
            cols.add(col);
            diag1.add(d1);
            diag2.add(d2);
            
            backtrack(row + 1);
            
            // Backtrack
            board[row][col] = '.';
            cols.delete(col);
            diag1.delete(d1);
            diag2.delete(d2);
        }
    }
    
    backtrack(0);
    return result;
}
```

## Dry Run

**Input:** n = 4

```
Initial board:
. . . .
. . . .
. . . .
. . . .

Row 0: Try placing queen in each column
- Col 0: Place Q, check conflicts (none), recurse to row 1
  Row 1: Try placing queen
    - Col 0: Conflict (same column)
    - Col 1: Conflict (diagonal)
    - Col 2: Place Q, check conflicts (none), recurse to row 2
      Row 2: Try placing queen
        - Col 0: Conflict (diagonal)
        - Col 1: Conflict (diagonal)
        - Col 2: Conflict (same column)
        - Col 3: Place Q, check conflicts (none), recurse to row 3
          Row 3: Try placing queen
            - Col 0: Conflict (diagonal)
            - Col 1: Place Q, check conflicts (none)
            - Found solution: [".Q..", "...Q", "Q...", "..Q."]
            - Backtrack
        - Backtrack
      - Backtrack
    - Col 3: Place Q, check conflicts (none), recurse to row 2
      Row 2: Try placing queen
        - Col 0: Place Q, check conflicts (none), recurse to row 3
          Row 3: Try placing queen
            - Col 0: Conflict (same column)
            - Col 1: Conflict (diagonal)
            - Col 2: Place Q, check conflicts (none)
            - Found solution: ["..Q.", "Q...", "...Q", ".Q.."]
            - Backtrack
        - Backtrack
      - Backtrack
  - Backtrack
- Col 1: Place Q, check conflicts (none), recurse to row 1
  ... (similar process)
- Col 2: Place Q, check conflicts (none), recurse to row 1
  ... (similar process)
- Col 3: Place Q, check conflicts (none), recurse to row 1
  ... (similar process)

Result: [".Q..", "...Q", "Q...", "..Q."], ["..Q.", "Q...", "...Q", ".Q.."]
```

## Complexity Analysis

- **Time Complexity:** O(n!) - In the worst case, we try all permutations
- **Space Complexity:** O(n²) - For the board and recursion stack

## Alternative Solutions

### Count Solutions Only
```javascript
function totalNQueens(n) {
    let count = 0;
    const cols = new Set();
    const diag1 = new Set();
    const diag2 = new Set();
    
    function backtrack(row) {
        if (row === n) {
            count++;
            return;
        }
        
        for (let col = 0; col < n; col++) {
            const d1 = row - col;
            const d2 = row + col;
            
            if (cols.has(col) || diag1.has(d1) || diag2.has(d2)) {
                continue;
            }
            
            cols.add(col);
            diag1.add(d1);
            diag2.add(d2);
            
            backtrack(row + 1);
            
            cols.delete(col);
            diag1.delete(d1);
            diag2.delete(d2);
        }
    }
    
    backtrack(0);
    return count;
}
```

### Bit Manipulation Approach
```javascript
function solveNQueensBitwise(n) {
    const result = [];
    
    function backtrack(row, cols, diag1, diag2, board) {
        if (row === n) {
            result.push(board.map(row => row.join('')));
            return;
        }
        
        let available = ((1 << n) - 1) & (~(cols | diag1 | diag2));
        
        while (available) {
            const pos = available & (-available);
            const col = Math.log2(pos);
            
            board[row][col] = 'Q';
            
            backtrack(
                row + 1,
                cols | pos,
                (diag1 | pos) << 1,
                (diag2 | pos) >> 1,
                board
            );
            
            board[row][col] = '.';
            available &= (available - 1);
        }
    }
    
    const board = Array(n).fill().map(() => Array(n).fill('.'));
    backtrack(0, 0, 0, 0, board);
    return result;
}
```

## Test Cases

```javascript
// Test cases
console.log(solveNQueens(1)); // [["Q"]]
console.log(solveNQueens(2)); // []
console.log(solveNQueens(3)); // []
console.log(solveNQueens(4)); // [["..Q.","Q...","...Q",".Q.."],[".Q..","...Q","Q...","..Q."]]
console.log(solveNQueens(5).length); // 10

// Count solutions
console.log(totalNQueens(4)); // 2
console.log(totalNQueens(8)); // 92
```

## Key Insights

1. **Constraint Satisfaction**: Each queen must not attack any other queen
2. **Row-by-Row Placement**: Place queens one row at a time to avoid row conflicts
3. **Conflict Detection**: Check column and diagonal conflicts efficiently
4. **Backtracking**: Undo placement when no valid solution exists
5. **Optimization**: Use sets or bit manipulation for faster conflict checking

## Related Problems

- [N-Queens II](NQueensII.md/) - Count solutions only
- [Sudoku Solver](SudokuSolver.md/) - Similar constraint satisfaction
- [Word Search](WordSearch.md/) - Backtracking in 2D grid
- [Generate Parentheses](GenerateParentheses.md/) - Backtracking with constraints
