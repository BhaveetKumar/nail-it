# Clone Graph

### Problem
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

**Example:**
```
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
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
    return cloneHelper(node, visited)
}

func cloneHelper(node *Node, visited map[*Node]*Node) *Node {
    if visited[node] != nil {
        return visited[node]
    }
    
    clone := &Node{Val: node.Val, Neighbors: make([]*Node, 0)}
    visited[node] = clone
    
    for _, neighbor := range node.Neighbors {
        clone.Neighbors = append(clone.Neighbors, cloneHelper(neighbor, visited))
    }
    
    return clone
}
```

### Complexity
- **Time Complexity:** O(V + E)
- **Space Complexity:** O(V)
