# Pacific Atlantic Water Flow

### Problem
There is an `m x n` rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an `m x n` integer matrix `heights` where `heights[r][c]` represents the height above sea level of the cell at coordinate `(r, c)`.

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates `result` where `result[i] = [ri, ci]` denotes that rain water can flow from cell `(ri, ci)` to both the Pacific and Atlantic oceans.

**Example:**
```
Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
```

### Golang Solution

```go
func pacificAtlantic(heights [][]int) [][]int {
    if len(heights) == 0 || len(heights[0]) == 0 {
        return [][]int{}
    }
    
    m, n := len(heights), len(heights[0])
    pacific := make([][]bool, m)
    atlantic := make([][]bool, m)
    
    for i := 0; i < m; i++ {
        pacific[i] = make([]bool, n)
        atlantic[i] = make([]bool, n)
    }
    
    // DFS from Pacific (top and left edges)
    for i := 0; i < m; i++ {
        dfs(heights, pacific, i, 0, m, n)
    }
    for j := 0; j < n; j++ {
        dfs(heights, pacific, 0, j, m, n)
    }
    
    // DFS from Atlantic (bottom and right edges)
    for i := 0; i < m; i++ {
        dfs(heights, atlantic, i, n-1, m, n)
    }
    for j := 0; j < n; j++ {
        dfs(heights, atlantic, m-1, j, m, n)
    }
    
    // Find cells that can reach both oceans
    var result [][]int
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if pacific[i][j] && atlantic[i][j] {
                result = append(result, []int{i, j})
            }
        }
    }
    
    return result
}

func dfs(heights [][]int, visited [][]bool, i, j, m, n int) {
    if visited[i][j] {
        return
    }
    
    visited[i][j] = true
    
    directions := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
    
    for _, dir := range directions {
        ni, nj := i+dir[0], j+dir[1]
        if ni >= 0 && ni < m && nj >= 0 && nj < n && heights[ni][nj] >= heights[i][j] {
            dfs(heights, visited, ni, nj, m, n)
        }
    }
}
```

### Complexity
- **Time Complexity:** O(m × n)
- **Space Complexity:** O(m × n)
