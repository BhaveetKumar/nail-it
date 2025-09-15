# Number of Islands

### Problem
Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Example:**
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Constraints:**
- m == grid.length
- n == grid[i].length
- 1 ≤ m, n ≤ 300
- grid[i][j] is '0' or '1'

### Explanation

#### **DFS Approach**
- Traverse the grid and when we find a '1', start DFS
- Mark all connected '1's as visited (change to '0')
- Count the number of times we start DFS
- Time Complexity: O(m × n)
- Space Complexity: O(m × n) for recursion stack

#### **BFS Approach**
- Use queue to explore all connected '1's
- Mark visited cells to avoid revisiting
- Time Complexity: O(m × n)
- Space Complexity: O(min(m, n)) for queue

#### **Union-Find Approach**
- Treat each '1' as a separate component
- Union adjacent '1's
- Count the number of components
- Time Complexity: O(m × n × α(m × n))
- Space Complexity: O(m × n)

### Dry Run

**Input:** `grid = [["1","1","1","1","0"], ["1","1","0","1","0"], ["1","1","0","0","0"], ["0","0","0","0","0"]]`

#### **DFS Approach**

| Step | Action | Grid State |
|------|--------|------------|
| 1 | Start at (0,0), found '1' | Start DFS |
| 2 | DFS from (0,0) | Mark all connected '1's as '0' |
| 3 | Grid becomes all '0's | Count = 1 |

**Result:** `1`

### Complexity
- **Time Complexity:** O(m × n) - Visit each cell once
- **Space Complexity:** O(m × n) - Recursion stack in worst case

### Golang Solution

#### **DFS Solution**
```go
func numIslands(grid [][]byte) int {
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }
    
    m, n := len(grid), len(grid[0])
    count := 0
    
    // Traverse the grid
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                // Found an island, start DFS
                dfs(grid, i, j, m, n)
                count++
            }
        }
    }
    
    return count
}

func dfs(grid [][]byte, i, j, m, n int) {
    // Check bounds and if current cell is land
    if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == '0' {
        return
    }
    
    // Mark current cell as visited
    grid[i][j] = '0'
    
    // Explore all four directions
    dfs(grid, i+1, j, m, n) // down
    dfs(grid, i-1, j, m, n) // up
    dfs(grid, i, j+1, m, n) // right
    dfs(grid, i, j-1, m, n) // left
}
```

#### **BFS Solution**
```go
func numIslands(grid [][]byte) int {
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }
    
    m, n := len(grid), len(grid[0])
    count := 0
    
    // Directions: up, down, left, right
    directions := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                // Found an island, start BFS
                queue := [][]int{{i, j}}
                grid[i][j] = '0' // Mark as visited
                
                for len(queue) > 0 {
                    cell := queue[0]
                    queue = queue[1:]
                    
                    // Explore all four directions
                    for _, dir := range directions {
                        newI, newJ := cell[0]+dir[0], cell[1]+dir[1]
                        
                        if newI >= 0 && newI < m && newJ >= 0 && newJ < n && grid[newI][newJ] == '1' {
                            grid[newI][newJ] = '0' // Mark as visited
                            queue = append(queue, []int{newI, newJ})
                        }
                    }
                }
                count++
            }
        }
    }
    
    return count
}
```

### Alternative Solutions

#### **Union-Find Solution**
```go
type UnionFind struct {
    parent []int
    rank   []int
    count  int
}

func NewUnionFind(n int) *UnionFind {
    parent := make([]int, n)
    rank := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    return &UnionFind{parent, rank, n}
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x])
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
    px, py := uf.Find(x), uf.Find(y)
    if px == py {
        return
    }
    
    if uf.rank[px] < uf.rank[py] {
        uf.parent[px] = py
    } else if uf.rank[px] > uf.rank[py] {
        uf.parent[py] = px
    } else {
        uf.parent[py] = px
        uf.rank[px]++
    }
    uf.count--
}

func numIslands(grid [][]byte) int {
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }
    
    m, n := len(grid), len(grid[0])
    uf := NewUnionFind(m * n)
    
    // Count initial number of '1's
    ones := 0
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                ones++
            }
        }
    }
    
    uf.count = ones
    
    // Union adjacent '1's
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                // Check right neighbor
                if j+1 < n && grid[i][j+1] == '1' {
                    uf.Union(i*n+j, i*n+j+1)
                }
                // Check down neighbor
                if i+1 < m && grid[i+1][j] == '1' {
                    uf.Union(i*n+j, (i+1)*n+j)
                }
            }
        }
    }
    
    return uf.count
}
```

#### **Iterative DFS with Stack**
```go
func numIslands(grid [][]byte) int {
    if len(grid) == 0 || len(grid[0]) == 0 {
        return 0
    }
    
    m, n := len(grid), len(grid[0])
    count := 0
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1' {
                // Use stack for iterative DFS
                stack := [][]int{{i, j}}
                
                for len(stack) > 0 {
                    cell := stack[len(stack)-1]
                    stack = stack[:len(stack)-1]
                    
                    row, col := cell[0], cell[1]
                    
                    if row < 0 || row >= m || col < 0 || col >= n || grid[row][col] == '0' {
                        continue
                    }
                    
                    grid[row][col] = '0'
                    
                    // Add all four directions to stack
                    stack = append(stack, []int{row+1, col})
                    stack = append(stack, []int{row-1, col})
                    stack = append(stack, []int{row, col+1})
                    stack = append(stack, []int{row, col-1})
                }
                count++
            }
        }
    }
    
    return count
}
```

### Notes / Variations

#### **Related Problems**
- **Max Area of Island**: Find the maximum area of an island
- **Number of Islands II**: Dynamic island creation
- **Surrounded Regions**: Mark regions surrounded by 'X'
- **Walls and Gates**: Fill empty rooms with distance to nearest gate
- **Pacific Atlantic Water Flow**: Find cells that can reach both oceans

#### **ICPC Insights**
- **Grid to Graph**: Convert 2D grid to graph representation
- **Connected Components**: Use DFS/BFS to find connected components
- **Memory Optimization**: Use iterative DFS to avoid stack overflow
- **Union-Find**: Efficient for dynamic connectivity problems

#### **Go-Specific Optimizations**
```go
// Use byte slice for better performance
func numIslands(grid [][]byte) int {
    // ... implementation
}

// Pre-allocate directions array
var directions = [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}

// Use struct for coordinates
type Point struct {
    row, col int
}
```

#### **Real-World Applications**
- **Image Processing**: Find connected components in binary images
- **Game Development**: Detect connected regions in game maps
- **Network Analysis**: Find connected components in networks
- **Geographic Information Systems**: Analyze land masses

### Testing

```go
func TestNumIslands(t *testing.T) {
    tests := []struct {
        grid     [][]byte
        expected int
    }{
        {
            [][]byte{
                {'1', '1', '1', '1', '0'},
                {'1', '1', '0', '1', '0'},
                {'1', '1', '0', '0', '0'},
                {'0', '0', '0', '0', '0'},
            },
            1,
        },
        {
            [][]byte{
                {'1', '1', '0', '0', '0'},
                {'1', '1', '0', '0', '0'},
                {'0', '0', '1', '0', '0'},
                {'0', '0', '0', '1', '1'},
            },
            3,
        },
        {
            [][]byte{
                {'1', '1', '1'},
                {'0', '1', '0'},
                {'1', '1', '1'},
            },
            1,
        },
    }
    
    for _, test := range tests {
        result := numIslands(test.grid)
        if result != test.expected {
            t.Errorf("numIslands(%v) = %d, expected %d", 
                test.grid, result, test.expected)
        }
    }
}
```

### Visualization

```
Input Grid:
1 1 1 1 0
1 1 0 1 0
1 1 0 0 0
0 0 0 0 0

DFS Traversal:
Step 1: Start at (0,0), found '1'
Step 2: DFS explores all connected '1's
Step 3: Mark all connected cells as '0'

After DFS:
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0

Result: 1 island found
```

### Performance Comparison

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| DFS | O(m×n) | O(m×n) | Simple, intuitive | Stack overflow risk |
| BFS | O(m×n) | O(min(m,n)) | Better space complexity | More complex |
| Union-Find | O(m×n×α) | O(m×n) | Good for dynamic problems | Overhead for static |
| Iterative DFS | O(m×n) | O(m×n) | No recursion limit | More memory |

**Recommendation**: Use DFS for simplicity, BFS for space optimization, Union-Find for dynamic connectivity problems.
