---
# Auto-generated front matter
Title: Numberofconnectedcomponentsinundirectedgraph
LastUpdated: 2025-11-06T20:45:58.744814
Tags: []
Status: draft
---

# Number of Connected Components in an Undirected Graph

### Problem
You have a graph of `n` nodes. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi`.

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
    parent := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    
    for _, edge := range edges {
        union(parent, edge[0], edge[1])
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

func union(parent []int, x, y int) {
    px, py := find(parent, x), find(parent, y)
    if px != py {
        parent[px] = py
    }
}
```

### Alternative Solutions

#### **DFS Approach**
```go
func countComponentsDFS(n int, edges [][]int) int {
    graph := make(map[int][]int)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }
    
    visited := make([]bool, n)
    count := 0
    
    for i := 0; i < n; i++ {
        if !visited[i] {
            dfs(graph, i, visited)
            count++
        }
    }
    
    return count
}

func dfs(graph map[int][]int, node int, visited []bool) {
    visited[node] = true
    
    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            dfs(graph, neighbor, visited)
        }
    }
}
```

#### **BFS Approach**
```go
func countComponentsBFS(n int, edges [][]int) int {
    graph := make(map[int][]int)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }
    
    visited := make([]bool, n)
    count := 0
    
    for i := 0; i < n; i++ {
        if !visited[i] {
            bfs(graph, i, visited)
            count++
        }
    }
    
    return count
}

func bfs(graph map[int][]int, start int, visited []bool) {
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

### Complexity
- **Time Complexity:** O(n + m) where m is number of edges
- **Space Complexity:** O(n)
