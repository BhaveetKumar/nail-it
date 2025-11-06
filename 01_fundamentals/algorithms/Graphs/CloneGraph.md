---
# Auto-generated front matter
Title: Clonegraph
LastUpdated: 2025-11-06T20:45:58.739134
Tags: []
Status: draft
---

# Clone Graph

### Problem
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

**Example:**
```
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val=1)'s neighbors are 2nd node (val=2) and 4th node (val=4).
2nd node (val=2)'s neighbors are 1st node (val=1) and 3rd node (val=3).
3rd node (val=3)'s neighbors are 2nd node (val=2) and 4th node (val=4).
4th node (val=4)'s neighbors are 1st node (val=1) and 3rd node (val=3).
```

### Golang Solution

```go
type Node struct {
    Val       int
    Neighbors []*Node
}

func cloneGraph(node *Node) *Node {
    if node == nil {
        return nil
    }
    
    visited := make(map[*Node]*Node)
    return cloneGraphHelper(node, visited)
}

func cloneGraphHelper(node *Node, visited map[*Node]*Node) *Node {
    if cloned, exists := visited[node]; exists {
        return cloned
    }
    
    cloned := &Node{Val: node.Val, Neighbors: []*Node{}}
    visited[node] = cloned
    
    for _, neighbor := range node.Neighbors {
        cloned.Neighbors = append(cloned.Neighbors, cloneGraphHelper(neighbor, visited))
    }
    
    return cloned
}
```

### Alternative Solutions

#### **Iterative DFS**
```go
func cloneGraphIterative(node *Node) *Node {
    if node == nil {
        return nil
    }
    
    visited := make(map[*Node]*Node)
    stack := []*Node{node}
    
    // Create first node
    cloned := &Node{Val: node.Val, Neighbors: []*Node{}}
    visited[node] = cloned
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        for _, neighbor := range current.Neighbors {
            if _, exists := visited[neighbor]; !exists {
                neighborClone := &Node{Val: neighbor.Val, Neighbors: []*Node{}}
                visited[neighbor] = neighborClone
                stack = append(stack, neighbor)
            }
            
            visited[current].Neighbors = append(visited[current].Neighbors, visited[neighbor])
        }
    }
    
    return cloned
}
```

#### **BFS Approach**
```go
func cloneGraphBFS(node *Node) *Node {
    if node == nil {
        return nil
    }
    
    visited := make(map[*Node]*Node)
    queue := []*Node{node}
    
    // Create first node
    cloned := &Node{Val: node.Val, Neighbors: []*Node{}}
    visited[node] = cloned
    
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        
        for _, neighbor := range current.Neighbors {
            if _, exists := visited[neighbor]; !exists {
                neighborClone := &Node{Val: neighbor.Val, Neighbors: []*Node{}}
                visited[neighbor] = neighborClone
                queue = append(queue, neighbor)
            }
            
            visited[current].Neighbors = append(visited[current].Neighbors, visited[neighbor])
        }
    }
    
    return cloned
}
```

#### **Using Array Index**
```go
func cloneGraphArray(node *Node) *Node {
    if node == nil {
        return nil
    }
    
    // First pass: collect all nodes
    nodes := collectNodes(node)
    
    // Create mapping from original to cloned
    cloned := make([]*Node, len(nodes))
    for i, n := range nodes {
        cloned[i] = &Node{Val: n.Val, Neighbors: []*Node{}}
    }
    
    // Second pass: connect neighbors
    for i, original := range nodes {
        for _, neighbor := range original.Neighbors {
            neighborIndex := findNodeIndex(nodes, neighbor)
            cloned[i].Neighbors = append(cloned[i].Neighbors, cloned[neighborIndex])
        }
    }
    
    return cloned[0]
}

func collectNodes(node *Node) []*Node {
    visited := make(map[*Node]bool)
    var nodes []*Node
    
    var dfs func(*Node)
    dfs = func(n *Node) {
        if visited[n] {
            return
        }
        
        visited[n] = true
        nodes = append(nodes, n)
        
        for _, neighbor := range n.Neighbors {
            dfs(neighbor)
        }
    }
    
    dfs(node)
    return nodes
}

func findNodeIndex(nodes []*Node, target *Node) int {
    for i, node := range nodes {
        if node == target {
            return i
        }
    }
    return -1
}
```

#### **Return Clone Info**
```go
type CloneInfo struct {
    Original *Node
    Cloned   *Node
    NodeCount int
}

func cloneGraphWithInfo(node *Node) CloneInfo {
    if node == nil {
        return CloneInfo{NodeCount: 0}
    }
    
    visited := make(map[*Node]*Node)
    cloned := cloneGraphHelper(node, visited)
    
    return CloneInfo{
        Original:  node,
        Cloned:    cloned,
        NodeCount: len(visited),
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(n) for the visited map and recursion stack