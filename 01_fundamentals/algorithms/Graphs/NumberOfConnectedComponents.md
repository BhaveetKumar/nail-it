---
# Auto-generated front matter
Title: Numberofconnectedcomponents
LastUpdated: 2025-11-06T20:45:58.738634
Tags: []
Status: draft
---

# Number of Connected Components in an Undirected Graph

### Problem

You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.

Return the number of connected components in the graph.

**Example:**

```
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2

Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
Output: 1
```

### Golang Solution

```go
func countComponents(n int, edges [][]int) int {
    // Build adjacency list
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    visited := make([]bool, n)
    components := 0

    for i := 0; i < n; i++ {
        if !visited[i] {
            dfs(graph, i, visited)
            components++
        }
    }

    return components
}

func dfs(graph [][]int, node int, visited []bool) {
    visited[node] = true

    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            dfs(graph, neighbor, visited)
        }
    }
}
```

### Alternative Solutions

#### **Using Union-Find**

```go
func countComponentsUnionFind(n int, edges [][]int) int {
    parent := make([]int, n)
    rank := make([]int, n)

    // Initialize
    for i := 0; i < n; i++ {
        parent[i] = i
        rank[i] = 0
    }

    // Union edges
    for _, edge := range edges {
        union(parent, rank, edge[0], edge[1])
    }

    // Count unique roots
    roots := make(map[int]bool)
    for i := 0; i < n; i++ {
        roots[find(parent, i)] = true
    }

    return len(roots)
}

func find(parent []int, x int) int {
    if parent[x] != x {
        parent[x] = find(parent, parent[x])
    }
    return parent[x]
}

func union(parent, rank []int, x, y int) {
    rootX := find(parent, x)
    rootY := find(parent, y)

    if rootX != rootY {
        if rank[rootX] < rank[rootY] {
            parent[rootX] = rootY
        } else if rank[rootX] > rank[rootY] {
            parent[rootY] = rootX
        } else {
            parent[rootY] = rootX
            rank[rootX]++
        }
    }
}
```

#### **Using BFS**

```go
func countComponentsBFS(n int, edges [][]int) int {
    // Build adjacency list
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    visited := make([]bool, n)
    components := 0

    for i := 0; i < n; i++ {
        if !visited[i] {
            bfs(graph, i, visited)
            components++
        }
    }

    return components
}

func bfs(graph [][]int, start int, visited []bool) {
    queue := []int{start}
    visited[start] = true

    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]

        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                visited[neighbor] = true
                queue = append(queue, neighbor)
            }
        }
    }
}
```

#### **Return Component Details**

```go
type ComponentInfo struct {
    Count       int
    Components  [][]int
    Sizes       []int
    LargestSize int
    SmallestSize int
}

func countComponentsWithDetails(n int, edges [][]int) ComponentInfo {
    // Build adjacency list
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    visited := make([]bool, n)
    var components [][]int
    var sizes []int

    for i := 0; i < n; i++ {
        if !visited[i] {
            component := []int{}
            size := dfsWithComponent(graph, i, visited, &component)
            components = append(components, component)
            sizes = append(sizes, size)
        }
    }

    largestSize := 0
    smallestSize := n

    for _, size := range sizes {
        if size > largestSize {
            largestSize = size
        }
        if size < smallestSize {
            smallestSize = size
        }
    }

    return ComponentInfo{
        Count:        len(components),
        Components:   components,
        Sizes:        sizes,
        LargestSize:  largestSize,
        SmallestSize: smallestSize,
    }
}

func dfsWithComponent(graph [][]int, node int, visited []bool, component *[]int) int {
    visited[node] = true
    *component = append(*component, node)
    size := 1

    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            size += dfsWithComponent(graph, neighbor, visited, component)
        }
    }

    return size
}
```

#### **Return Connected Pairs**

```go
func findConnectedPairs(n int, edges [][]int) [][]int {
    // Build adjacency list
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    visited := make([]bool, n)
    var pairs [][]int

    for i := 0; i < n; i++ {
        if !visited[i] {
            component := []int{}
            dfsWithComponent(graph, i, visited, &component)

            // Add all pairs within component
            for j := 0; j < len(component); j++ {
                for k := j + 1; k < len(component); k++ {
                    pairs = append(pairs, []int{component[j], component[k]})
                }
            }
        }
    }

    return pairs
}
```

#### **Return Component Statistics**

```go
type ComponentStats struct {
    TotalComponents int
    TotalNodes      int
    TotalEdges      int
    AvgComponentSize float64
    MaxComponentSize int
    MinComponentSize int
    IsolatedNodes   int
}

func componentStatistics(n int, edges [][]int) ComponentStats {
    // Build adjacency list
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    visited := make([]bool, n)
    var sizes []int
    isolatedNodes := 0

    for i := 0; i < n; i++ {
        if !visited[i] {
            component := []int{}
            size := dfsWithComponent(graph, i, visited, &component)
            sizes = append(sizes, size)

            if size == 1 {
                isolatedNodes++
            }
        }
    }

    totalSize := 0
    maxSize := 0
    minSize := n

    for _, size := range sizes {
        totalSize += size
        if size > maxSize {
            maxSize = size
        }
        if size < minSize {
            minSize = size
        }
    }

    return ComponentStats{
        TotalComponents:  len(sizes),
        TotalNodes:       n,
        TotalEdges:       len(edges),
        AvgComponentSize: float64(totalSize) / float64(len(sizes)),
        MaxComponentSize: maxSize,
        MinComponentSize: minSize,
        IsolatedNodes:    isolatedNodes,
    }
}
```

### Complexity

- **Time Complexity:** O(V + E) where V is vertices and E is edges
- **Space Complexity:** O(V + E) for adjacency list, O(V) for visited array
