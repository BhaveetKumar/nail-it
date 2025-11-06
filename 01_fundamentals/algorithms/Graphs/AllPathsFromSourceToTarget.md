---
# Auto-generated front matter
Title: Allpathsfromsourcetotarget
LastUpdated: 2025-11-06T20:45:58.744284
Tags: []
Status: draft
---

# All Paths From Source to Target

### Problem
Given a directed acyclic graph (DAG) of `n` nodes labeled from `0` to `n - 1`, find all possible paths from node `0` to node `n - 1` and return them in any order.

The graph is given as follows: `graph[i]` is a list of all nodes you can visit from node `i` (i.e., there is a directed edge from node `i` to node `graph[i][j]`).

**Example:**
```
Input: graph = [[1,2],[3],[3],[]]
Output: [[0,1,3],[0,2,3]]
Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.

Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
```

### Golang Solution

```go
func allPathsSourceTarget(graph [][]int) [][]int {
    var result [][]int
    var currentPath []int
    
    var dfs func(int)
    dfs = func(node int) {
        currentPath = append(currentPath, node)
        
        if node == len(graph)-1 {
            path := make([]int, len(currentPath))
            copy(path, currentPath)
            result = append(result, path)
        } else {
            for _, neighbor := range graph[node] {
                dfs(neighbor)
            }
        }
        
        currentPath = currentPath[:len(currentPath)-1]
    }
    
    dfs(0)
    return result
}
```

### Alternative Solutions

#### **Iterative DFS**
```go
func allPathsSourceTargetIterative(graph [][]int) [][]int {
    var result [][]int
    stack := [][]int{{0}}
    
    for len(stack) > 0 {
        currentPath := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        lastNode := currentPath[len(currentPath)-1]
        
        if lastNode == len(graph)-1 {
            result = append(result, currentPath)
        } else {
            for _, neighbor := range graph[lastNode] {
                newPath := make([]int, len(currentPath))
                copy(newPath, currentPath)
                newPath = append(newPath, neighbor)
                stack = append(stack, newPath)
            }
        }
    }
    
    return result
}
```

#### **BFS Approach**
```go
func allPathsSourceTargetBFS(graph [][]int) [][]int {
    var result [][]int
    queue := [][]int{{0}}
    
    for len(queue) > 0 {
        currentPath := queue[0]
        queue = queue[1:]
        
        lastNode := currentPath[len(currentPath)-1]
        
        if lastNode == len(graph)-1 {
            result = append(result, currentPath)
        } else {
            for _, neighbor := range graph[lastNode] {
                newPath := make([]int, len(currentPath))
                copy(newPath, currentPath)
                newPath = append(newPath, neighbor)
                queue = append(queue, newPath)
            }
        }
    }
    
    return result
}
```

#### **With Memoization**
```go
func allPathsSourceTargetMemo(graph [][]int) [][]int {
    memo := make(map[int][][]int)
    return dfsWithMemo(graph, 0, memo)
}

func dfsWithMemo(graph [][]int, node int, memo map[int][][]int) [][]int {
    if node == len(graph)-1 {
        return [][]int{{node}}
    }
    
    if paths, exists := memo[node]; exists {
        return paths
    }
    
    var allPaths [][]int
    
    for _, neighbor := range graph[node] {
        neighborPaths := dfsWithMemo(graph, neighbor, memo)
        
        for _, path := range neighborPaths {
            newPath := make([]int, len(path)+1)
            newPath[0] = node
            copy(newPath[1:], path)
            allPaths = append(allPaths, newPath)
        }
    }
    
    memo[node] = allPaths
    return allPaths
}
```

### Complexity
- **Time Complexity:** O(2^n × n) in worst case
- **Space Complexity:** O(2^n × n)
