# N-Queens

### Problem
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

**Example:**
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
```

### Golang Solution

```go
func solveNQueens(n int) [][]string {
    var result [][]string
    board := make([]int, n) // board[i] = column of queen in row i
    
    var backtrack func(int)
    backtrack = func(row int) {
        if row == n {
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
        
        for col := 0; col < n; col++ {
            if isValid(board, row, col) {
                board[row] = col
                backtrack(row + 1)
            }
        }
    }
    
    backtrack(0)
    return result
}

func isValid(board []int, row, col int) bool {
    for i := 0; i < row; i++ {
        if board[i] == col || abs(board[i]-col) == abs(i-row) {
            return false
        }
    }
    return true
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```

### Complexity
- **Time Complexity:** O(N!)
- **Space Complexity:** O(N)
