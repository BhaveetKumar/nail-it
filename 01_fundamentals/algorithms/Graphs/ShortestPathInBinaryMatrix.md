# Shortest Path in Binary Matrix

### Problem
Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from the top-left corner (0, 0) to the bottom-right corner (n-1, n-1). If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n-1, n-1)) such that:

- All the visited cells of the path are 0.
- All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).

The length of a clear path is the number of visited cells of this path.

**Example:**
```
Input: grid = [[0,1],[1,0]]
Output: 2

Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
```

### Golang Solution

```go
func shortestPathBinaryMatrix(grid [][]int) int {
    n := len(grid)
    if grid[0][0] == 1 || grid[n-1][n-1] == 1 {
        return -1
    }
    
    if n == 1 {
        return 1
    }
    
    // 8 directions
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    queue := [][]int{{0, 0, 1}} // [row, col, distance]
    grid[0][0] = 1 // Mark as visited
    
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        
        row, col, dist := current[0], current[1], current[2]
        
        for _, dir := range directions {
            newRow, newCol := row+dir[0], col+dir[1]
            
            if newRow == n-1 && newCol == n-1 {
                return dist + 1
            }
            
            if newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && grid[newRow][newCol] == 0 {
                grid[newRow][newCol] = 1 // Mark as visited
                queue = append(queue, []int{newRow, newCol, dist + 1})
            }
        }
    }
    
    return -1
}
```

### Alternative Solutions

#### **DFS Approach**
```go
func shortestPathBinaryMatrixDFS(grid [][]int) int {
    n := len(grid)
    if grid[0][0] == 1 || grid[n-1][n-1] == 1 {
        return -1
    }
    
    visited := make([][]bool, n)
    for i := range visited {
        visited[i] = make([]bool, n)
    }
    
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    minPath := math.MaxInt32
    
    var dfs func(int, int, int)
    dfs = func(row, col, dist int) {
        if row == n-1 && col == n-1 {
            minPath = min(minPath, dist)
            return
        }
        
        for _, dir := range directions {
            newRow, newCol := row+dir[0], col+dir[1]
            
            if newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && 
               grid[newRow][newCol] == 0 && !visited[newRow][newCol] {
                visited[newRow][newCol] = true
                dfs(newRow, newCol, dist+1)
                visited[newRow][newCol] = false
            }
        }
    }
    
    visited[0][0] = true
    dfs(0, 0, 1)
    
    if minPath == math.MaxInt32 {
        return -1
    }
    return minPath
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

#### **A* Algorithm**
```go
type Node struct {
    row, col, dist int
    priority       int
}

type PriorityQueue []Node

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].priority < pq[j].priority }
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(Node))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    x := old[n-1]
    *pq = old[0 : n-1]
    return x
}

func shortestPathBinaryMatrixAStar(grid [][]int) int {
    n := len(grid)
    if grid[0][0] == 1 || grid[n-1][n-1] == 1 {
        return -1
    }
    
    if n == 1 {
        return 1
    }
    
    directions := [][]int{
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1},
    }
    
    pq := &PriorityQueue{}
    heap.Init(pq)
    heap.Push(pq, Node{0, 0, 1, 1 + heuristic(0, 0, n-1, n-1)})
    
    visited := make([][]bool, n)
    for i := range visited {
        visited[i] = make([]bool, n)
    }
    
    for pq.Len() > 0 {
        current := heap.Pop(pq).(Node)
        
        if current.row == n-1 && current.col == n-1 {
            return current.dist
        }
        
        if visited[current.row][current.col] {
            continue
        }
        
        visited[current.row][current.col] = true
        
        for _, dir := range directions {
            newRow, newCol := current.row+dir[0], current.col+dir[1]
            
            if newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && 
               grid[newRow][newCol] == 0 && !visited[newRow][newCol] {
                newDist := current.dist + 1
                priority := newDist + heuristic(newRow, newCol, n-1, n-1)
                heap.Push(pq, Node{newRow, newCol, newDist, priority})
            }
        }
    }
    
    return -1
}

func heuristic(row, col, targetRow, targetCol int) int {
    return max(abs(row-targetRow), abs(col-targetCol))
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n²) for BFS, O(8^(n²)) for DFS, O(n² log n) for A*
- **Space Complexity:** O(n²)
