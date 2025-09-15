# Word Search

### Problem
Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

**Example:**
```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true
```

### Golang Solution

```go
func exist(board [][]byte, word string) bool {
    m, n := len(board), len(board[0])
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if dfs(board, word, i, j, 0) {
                return true
            }
        }
    }
    
    return false
}

func dfs(board [][]byte, word string, i, j, index int) bool {
    if index == len(word) {
        return true
    }
    
    if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || board[i][j] != word[index] {
        return false
    }
    
    // Mark cell as visited
    temp := board[i][j]
    board[i][j] = '#'
    
    // Check all four directions
    directions := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
    for _, dir := range directions {
        if dfs(board, word, i+dir[0], j+dir[1], index+1) {
            return true
        }
    }
    
    // Restore cell
    board[i][j] = temp
    return false
}
```

### Complexity
- **Time Complexity:** O(m × n × 4^L) where L is word length
- **Space Complexity:** O(L) for recursion depth
