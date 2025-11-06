---
# Auto-generated front matter
Title: Graphvalidtree
LastUpdated: 2025-11-06T20:45:58.736995
Tags: []
Status: draft
---

# Graph Valid Tree

### Problem
You have a graph of `n` nodes labeled from `0` to `n - 1`. You are given an integer `n` and an array `edges` where `edges[i] = [ai, bi]` indicates that there is an undirected edge between nodes `ai` and `bi` in the graph.

Return `true` if the edges of the given graph make up a valid tree, and `false` otherwise.

**Example:**
```
Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: false

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false
```

### Golang Solution

```go
func validTree(n int, edges [][]int) bool {
    if len(edges) != n-1 {
        return false
    }
    
    // Build adjacency list
    graph := make(map[int][]int)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }
    
    // DFS to check connectivity and cycles
    visited := make(map[int]bool)
    
    var dfs func(int, int) bool
    dfs = func(node, parent int) bool {
        if visited[node] {
            return false // Cycle detected
        }
        
        visited[node] = true
        
        for _, neighbor := range graph[node] {
            if neighbor != parent {
                if !dfs(neighbor, node) {
                    return false
                }
            }
        }
        
        return true
    }
    
    if !dfs(0, -1) {
        return false
    }
    
    // Check if all nodes are visited
    return len(visited) == n
}
```

### Alternative Solutions

#### **Union-Find Approach**
```go
func validTreeUnionFind(n int, edges [][]int) bool {
    if len(edges) != n-1 {
        return false
    }
    
    parent := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    
    for _, edge := range edges {
        if !union(parent, edge[0], edge[1]) {
            return false // Cycle detected
        }
    }
    
    return true
}

func find(parent []int, x int) int {
    if parent[x] != x {
        parent[x] = find(parent, parent[x])
    }
    return parent[x]
}

func union(parent []int, x, y int) bool {
    px, py := find(parent, x), find(parent, y)
    if px == py {
        return false // Already connected, cycle detected
    }
    parent[px] = py
    return true
}
```

### Complexity
- **Time Complexity:** O(n + m) where m is number of edges
- **Space Complexity:** O(n)
