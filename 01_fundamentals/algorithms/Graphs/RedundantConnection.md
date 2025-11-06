---
# Auto-generated front matter
Title: Redundantconnection
LastUpdated: 2025-11-06T20:45:58.741058
Tags: []
Status: draft
---

# Redundant Connection

### Problem
In this problem, a tree is an undirected graph that is connected and has no cycles.

You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.

Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.

**Example:**
```
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]

Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
Output: [1,4]
```

### Golang Solution

```go
func findRedundantConnection(edges [][]int) []int {
    n := len(edges)
    parent := make([]int, n+1)
    
    // Initialize parent array
    for i := 1; i <= n; i++ {
        parent[i] = i
    }
    
    for _, edge := range edges {
        if find(parent, edge[0]) == find(parent, edge[1]) {
            return edge
        }
        union(parent, edge[0], edge[1])
    }
    
    return []int{}
}

func find(parent []int, x int) int {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]) // Path compression
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
func findRedundantConnectionDFS(edges [][]int) []int {
    n := len(edges)
    graph := make(map[int][]int)
    
    for _, edge := range edges {
        u, v := edge[0], edge[1]
        
        // Check if adding this edge creates a cycle
        if hasPath(graph, u, v, make(map[int]bool)) {
            return edge
        }
        
        // Add edge to graph
        graph[u] = append(graph[u], v)
        graph[v] = append(graph[v], u)
    }
    
    return []int{}
}

func hasPath(graph map[int][]int, start, end int, visited map[int]bool) bool {
    if start == end {
        return true
    }
    
    visited[start] = true
    
    for _, neighbor := range graph[start] {
        if !visited[neighbor] && hasPath(graph, neighbor, end, visited) {
            return true
        }
    }
    
    return false
}
```

### Complexity
- **Time Complexity:** O(n α(n)) for Union-Find, O(n²) for DFS
- **Space Complexity:** O(n)
