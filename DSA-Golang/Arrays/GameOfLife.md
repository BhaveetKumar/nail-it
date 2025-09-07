# Game of Life

### Problem
According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an `m x n` grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules:

1. Any live cell with fewer than two live neighbors dies as if caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population.
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

**Example:**
```
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
```

### Golang Solution

```go
func gameOfLife(board [][]int) {
    m, n := len(board), len(board[0])
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            liveNeighbors := 0
            
            for _, dir := range directions {
                ni, nj := i+dir[0], j+dir[1]
                if ni >= 0 && ni < m && nj >= 0 && nj < n {
                    if board[ni][nj] == 1 || board[ni][nj] == -1 {
                        liveNeighbors++
                    }
                }
            }
            
            if board[i][j] == 1 {
                if liveNeighbors < 2 || liveNeighbors > 3 {
                    board[i][j] = -1 // Live to dead
                }
            } else {
                if liveNeighbors == 3 {
                    board[i][j] = 2 // Dead to live
                }
            }
        }
    }
    
    // Update the board
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if board[i][j] == -1 {
                board[i][j] = 0
            } else if board[i][j] == 2 {
                board[i][j] = 1
            }
        }
    }
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1)
