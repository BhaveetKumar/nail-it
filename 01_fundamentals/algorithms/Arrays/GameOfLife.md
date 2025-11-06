---
# Auto-generated front matter
Title: Gameoflife
LastUpdated: 2025-11-06T20:45:58.727047
Tags: []
Status: draft
---

# Game of Life

### Problem
According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an `m x n` grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules:

1. Any live cell with fewer than two live neighbors dies as if caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population.
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously.

**Example:**
```
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]
```

### Golang Solution

```go
func gameOfLife(board [][]int) {
    m, n := len(board), len(board[0])
    
    // Directions for 8 neighbors
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    // First pass: mark cells with special values
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            liveNeighbors := 0
            
            // Count live neighbors
            for _, dir := range directions {
                ni, nj := i+dir[0], j+dir[1]
                if ni >= 0 && ni < m && nj >= 0 && nj < n {
                    if board[ni][nj] == 1 || board[ni][nj] == -1 {
                        liveNeighbors++
                    }
                }
            }
            
            // Apply rules
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
    
    // Second pass: convert special values to 0 and 1
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

### Alternative Solutions

#### **Using Extra Space**
```go
func gameOfLifeExtraSpace(board [][]int) {
    m, n := len(board), len(board[0])
    next := make([][]int, m)
    for i := range next {
        next[i] = make([]int, n)
    }
    
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
                    if board[ni][nj] == 1 {
                        liveNeighbors++
                    }
                }
            }
            
            if board[i][j] == 1 {
                if liveNeighbors == 2 || liveNeighbors == 3 {
                    next[i][j] = 1
                }
            } else {
                if liveNeighbors == 3 {
                    next[i][j] = 1
                }
            }
        }
    }
    
    // Copy next to board
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            board[i][j] = next[i][j]
        }
    }
}
```

#### **Functional Approach**
```go
func gameOfLifeFunctional(board [][]int) {
    m, n := len(board), len(board[0])
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            liveNeighbors := countLiveNeighbors(board, i, j, m, n)
            board[i][j] = getNextState(board[i][j], liveNeighbors)
        }
    }
    
    // Convert back to 0/1
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

func countLiveNeighbors(board [][]int, i, j, m, n int) int {
    count := 0
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    for _, dir := range directions {
        ni, nj := i+dir[0], j+dir[1]
        if ni >= 0 && ni < m && nj >= 0 && nj < n {
            if board[ni][nj] == 1 || board[ni][nj] == -1 {
                count++
            }
        }
    }
    
    return count
}

func getNextState(current, liveNeighbors int) int {
    if current == 1 {
        if liveNeighbors < 2 || liveNeighbors > 3 {
            return -1 // Live to dead
        }
        return 1 // Stay alive
    } else {
        if liveNeighbors == 3 {
            return 2 // Dead to live
        }
        return 0 // Stay dead
    }
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(1) for in-place, O(m × n) for extra space