# Graphs Pattern

> **Master graph algorithms with Go implementations**

## 📋 Problems

### **BFS (Breadth-First Search)**
- [Binary Tree Level Order Traversal](./BinaryTreeLevelOrderTraversal.md) - BFS on trees
- [Word Ladder](./WordLadder.md) - Shortest path in word graph
- [Rotting Oranges](./RottingOranges.md) - Multi-source BFS
- [Open the Lock](./OpenTheLock.md) - BFS with state transitions
- [Minimum Knight Moves](./MinimumKnightMoves.md) - BFS on chessboard

### **DFS (Depth-First Search)**
- [Number of Islands](./NumberOfIslands.md) - Connected components
- [Course Schedule](./CourseSchedule.md) - Cycle detection
- [Clone Graph](./CloneGraph.md) - Graph cloning
- [Pacific Atlantic Water Flow](./PacificAtlanticWaterFlow.md) - Multi-source DFS
- [All Paths From Source to Target](./AllPathsFromSourceToTarget.md) - Path finding

### **Shortest Path**
- [Network Delay Time](./NetworkDelayTime.md) - Dijkstra's algorithm
- [Cheapest Flights Within K Stops](./CheapestFlightsWithinKStops.md) - Modified Dijkstra
- [Path With Minimum Effort](./PathWithMinimumEffort.md) - Binary search + BFS
- [Swim in Rising Water](./SwimInRisingWater.md) - Union-Find + Binary search

### **Topological Sort**
- [Course Schedule II](./CourseScheduleII.md) - Topological ordering
- [Alien Dictionary](./AlienDictionary.md) - Topological sort on characters
- [Sequence Reconstruction](./SequenceReconstruction.md) - Unique topological order

### **Union-Find**
- [Number of Connected Components](./NumberOfConnectedComponents.md) - Union-Find basics
- [Redundant Connection](./RedundantConnection.md) - Cycle detection
- [Accounts Merge](./AccountsMerge.md) - Connected components
- [Most Stones Removed](./MostStonesRemoved.md) - Union-Find optimization

---

## 🎯 Key Concepts

### **Graph Representation**
- **Adjacency List**: `map[int][]int` or `[][]int`
- **Adjacency Matrix**: `[][]int` for dense graphs
- **Edge List**: `[][]int` for sparse graphs

### **Traversal Algorithms**
- **BFS**: Level-by-level traversal, shortest path in unweighted graphs
- **DFS**: Deep traversal, cycle detection, topological sort
- **Iterative DFS**: Use stack instead of recursion

### **Shortest Path Algorithms**
- **BFS**: Unweighted graphs, O(V + E)
- **Dijkstra**: Non-negative weights, O((V + E) log V)
- **Bellman-Ford**: Negative weights, O(VE)
- **Floyd-Warshall**: All pairs, O(V³)

---

## 🛠️ Go-Specific Tips

### **Graph Representation**
```go
// Adjacency list
type Graph struct {
    nodes map[int][]int
}

func NewGraph() *Graph {
    return &Graph{
        nodes: make(map[int][]int),
    }
}

func (g *Graph) AddEdge(from, to int) {
    g.nodes[from] = append(g.nodes[from], to)
}

// For undirected graphs
func (g *Graph) AddUndirectedEdge(from, to int) {
    g.AddEdge(from, to)
    g.AddEdge(to, from)
}
```

### **BFS Implementation**
```go
func bfs(graph map[int][]int, start int) []int {
    visited := make(map[int]bool)
    queue := []int{start}
    result := []int{}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        if visited[node] {
            continue
        }
        
        visited[node] = true
        result = append(result, node)
        
        for _, neighbor := range graph[node] {
            if !visited[neighbor] {
                queue = append(queue, neighbor)
            }
        }
    }
    
    return result
}
```

### **DFS Implementation**
```go
func dfs(graph map[int][]int, start int) []int {
    visited := make(map[int]bool)
    result := []int{}
    
    var dfsHelper func(int)
    dfsHelper = func(node int) {
        if visited[node] {
            return
        }
        
        visited[node] = true
        result = append(result, node)
        
        for _, neighbor := range graph[node] {
            dfsHelper(neighbor)
        }
    }
    
    dfsHelper(start)
    return result
}
```

### **Dijkstra's Algorithm**
```go
import (
    "container/heap"
    "math"
)

type Edge struct {
    To     int
    Weight int
}

type PriorityQueue []*Item

type Item struct {
    Node     int
    Distance int
    Index    int
}

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].Distance < pq[j].Distance }
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
    item := x.(*Item)
    item.Index = len(*pq)
    *pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    old[n-1] = nil
    item.Index = -1
    *pq = old[0 : n-1]
    return item
}

func dijkstra(graph map[int][]Edge, start int) map[int]int {
    distances := make(map[int]int)
    for node := range graph {
        distances[node] = math.MaxInt32
    }
    distances[start] = 0
    
    pq := &PriorityQueue{}
    heap.Init(pq)
    heap.Push(pq, &Item{Node: start, Distance: 0})
    
    for pq.Len() > 0 {
        item := heap.Pop(pq).(*Item)
        node := item.Node
        
        if item.Distance > distances[node] {
            continue
        }
        
        for _, edge := range graph[node] {
            newDist := distances[node] + edge.Weight
            if newDist < distances[edge.To] {
                distances[edge.To] = newDist
                heap.Push(pq, &Item{Node: edge.To, Distance: newDist})
            }
        }
    }
    
    return distances
}
```

### **Union-Find Implementation**
```go
type UnionFind struct {
    parent []int
    rank   []int
}

func NewUnionFind(n int) *UnionFind {
    parent := make([]int, n)
    rank := make([]int, n)
    for i := range parent {
        parent[i] = i
    }
    return &UnionFind{parent, rank}
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x]) // Path compression
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
}
```

---

## 🎯 Interview Tips

### **How to Identify Graph Problems**
1. **Connected Components**: Use DFS or Union-Find
2. **Shortest Path**: Use BFS (unweighted) or Dijkstra (weighted)
3. **Cycle Detection**: Use DFS with visited states
4. **Topological Sort**: Use DFS or Kahn's algorithm
5. **Minimum Spanning Tree**: Use Kruskal's or Prim's algorithm

### **Common Graph Problem Patterns**
- **Grid Problems**: Convert 2D grid to graph
- **State Space Search**: Each state is a node
- **Tree Problems**: Special case of graphs
- **Network Flow**: Maximum flow, minimum cut

### **Optimization Tips**
- **Use appropriate data structures**: Priority queue for Dijkstra
- **Avoid redundant computations**: Memoization for repeated subproblems
- **Space optimization**: Use iterative DFS for deep graphs
- **Early termination**: Stop when target is found
