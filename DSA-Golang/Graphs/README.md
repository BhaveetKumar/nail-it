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

**Detailed Explanation:**
Graph representation is fundamental to implementing graph algorithms efficiently. The choice of representation significantly impacts both time and space complexity, as well as the ease of implementation for different algorithms.

**Adjacency List:**

- **Structure**: Each vertex maintains a list of its adjacent vertices
- **Space Complexity**: O(V + E) where V is vertices and E is edges
- **Time Complexity**: O(degree(v)) to iterate through neighbors of vertex v
- **Best For**: Sparse graphs, most graph algorithms
- **Go Implementation**: `map[int][]int` or `[][]int`

**Adjacency Matrix:**

- **Structure**: 2D array where matrix[i][j] indicates edge between vertices i and j
- **Space Complexity**: O(V²) regardless of number of edges
- **Time Complexity**: O(1) to check if edge exists, O(V) to iterate through neighbors
- **Best For**: Dense graphs, algorithms requiring frequent edge existence checks
- **Go Implementation**: `[][]int` or `[][]bool`

**Edge List:**

- **Structure**: List of all edges as pairs of vertices
- **Space Complexity**: O(E) for storing edges
- **Time Complexity**: O(E) to iterate through all edges
- **Best For**: Algorithms that process all edges, sparse graphs
- **Go Implementation**: `[][]int` where each inner array represents an edge

**Discussion Questions & Answers:**

**Q1: When should you use adjacency list vs adjacency matrix for graph representation?**

**Answer:** Choose based on graph characteristics:

- **Adjacency List**: Use when graph is sparse (E << V²), need to iterate through neighbors frequently, memory is a constraint
- **Adjacency Matrix**: Use when graph is dense (E ≈ V²), need frequent edge existence checks, graph is small
- **Trade-offs**: Adjacency list saves space but slower edge existence checks; adjacency matrix is faster for edge checks but uses more space
- **Algorithm Considerations**: BFS/DFS work well with adjacency list; algorithms requiring edge weight lookups benefit from adjacency matrix

**Q2: How do you handle weighted graphs in different representations?**

**Answer:** Weighted graph representations:

- **Adjacency List**: Store edge weights in the list elements (structs or tuples)
- **Adjacency Matrix**: Store weights in matrix cells, use special values for non-existent edges
- **Edge List**: Include weight as third element in edge representation
- **Go Implementation**: Use structs for edges with weight fields
- **Memory Considerations**: Adjacency matrix for weighted graphs uses O(V²) space regardless of actual edges

### **Traversal Algorithms**

**Detailed Explanation:**
Graph traversal algorithms are fundamental building blocks for many graph problems. They provide systematic ways to visit all vertices and edges in a graph, forming the basis for more complex algorithms.

**BFS (Breadth-First Search):**

- **Strategy**: Explore vertices level by level, visiting all neighbors before moving to next level
- **Data Structure**: Queue (FIFO) to maintain order of exploration
- **Applications**: Shortest path in unweighted graphs, level-order traversal, connected components
- **Time Complexity**: O(V + E) for adjacency list, O(V²) for adjacency matrix
- **Space Complexity**: O(V) for queue and visited set
- **Key Insight**: Guarantees shortest path in unweighted graphs

**DFS (Depth-First Search):**

- **Strategy**: Explore as far as possible along each branch before backtracking
- **Data Structure**: Stack (LIFO) or recursion
- **Applications**: Cycle detection, topological sorting, path finding, connected components
- **Time Complexity**: O(V + E) for adjacency list, O(V²) for adjacency matrix
- **Space Complexity**: O(V) for recursion stack or explicit stack
- **Key Insight**: Natural for recursive problems and backtracking

**Iterative DFS:**

- **Implementation**: Use explicit stack instead of recursion
- **Advantages**: Avoids stack overflow for deep graphs, better control over traversal
- **Disadvantages**: More complex implementation, requires manual stack management
- **Use Cases**: Very deep graphs, when recursion depth is a concern

**Discussion Questions & Answers:**

**Q1: How do you choose between BFS and DFS for different graph problems?**

**Answer:** Choose based on problem requirements:

- **BFS**: Use for shortest path problems, level-order processing, when you need to explore all vertices at distance k before distance k+1
- **DFS**: Use for cycle detection, topological sorting, when you need to explore one path completely before trying others
- **Memory Considerations**: BFS uses more memory for wide graphs, DFS uses more memory for deep graphs
- **Performance**: BFS is better for finding shortest paths, DFS is better for exploring all possible paths

**Q2: What are the common pitfalls when implementing graph traversal algorithms?**

**Answer:** Common pitfalls include:

- **Infinite Loops**: Not properly marking vertices as visited
- **Stack Overflow**: Deep recursion in DFS without proper base cases
- **Memory Issues**: Not cleaning up visited sets or queues
- **Incorrect Order**: Wrong traversal order due to incorrect data structure usage
- **Edge Cases**: Not handling disconnected graphs or single-vertex graphs
- **Performance**: Using wrong data structures (e.g., slice for queue instead of proper queue)

### **Shortest Path Algorithms**

**Detailed Explanation:**
Shortest path algorithms find the minimum cost path between vertices in a graph. The choice of algorithm depends on graph characteristics like edge weights and graph structure.

**BFS for Shortest Path:**

- **Use Case**: Unweighted graphs or graphs with unit edge weights
- **Time Complexity**: O(V + E)
- **Space Complexity**: O(V)
- **Key Insight**: First time a vertex is visited, it's via the shortest path
- **Implementation**: Standard BFS with distance tracking

**Dijkstra's Algorithm:**

- **Use Case**: Graphs with non-negative edge weights
- **Time Complexity**: O((V + E) log V) with binary heap, O(V²) with array
- **Space Complexity**: O(V)
- **Key Insight**: Greedy algorithm that always processes the vertex with minimum distance
- **Data Structure**: Priority queue (min-heap) for efficient minimum extraction

**Bellman-Ford Algorithm:**

- **Use Case**: Graphs with negative edge weights, negative cycle detection
- **Time Complexity**: O(VE)
- **Space Complexity**: O(V)
- **Key Insight**: Relax all edges V-1 times, then check for negative cycles
- **Advantage**: Can handle negative weights and detect negative cycles

**Floyd-Warshall Algorithm:**

- **Use Case**: All-pairs shortest path problems
- **Time Complexity**: O(V³)
- **Space Complexity**: O(V²)
- **Key Insight**: Dynamic programming approach that considers all intermediate vertices
- **Advantage**: Finds shortest paths between all pairs of vertices

**Discussion Questions & Answers:**

**Q1: How do you choose the right shortest path algorithm for a given problem?**

**Answer:** Algorithm selection criteria:

- **BFS**: Unweighted graphs, unit weights, when you need shortest path in terms of number of edges
- **Dijkstra**: Non-negative weights, single-source shortest path, when you need optimal solution
- **Bellman-Ford**: Negative weights, when you need to detect negative cycles
- **Floyd-Warshall**: All-pairs shortest path, when you need distances between all vertex pairs
- **Performance**: Consider graph size, edge density, and weight characteristics

**Q2: What are the implementation challenges for Dijkstra's algorithm in Go?**

**Answer:** Implementation challenges:

- **Priority Queue**: Go doesn't have built-in priority queue, need to implement or use third-party library
- **Heap Operations**: Proper heap implementation with decrease-key operation
- **Memory Management**: Efficient handling of heap elements and distance updates
- **Edge Cases**: Handling disconnected graphs, unreachable vertices
- **Performance**: Optimizing heap operations for large graphs
- **Code Complexity**: Balancing readability with performance optimizations

---

## 🛠️ Go-Specific Tips

### **Graph Representation**

**Detailed Explanation:**
Go's type system and built-in data structures provide excellent support for implementing graph algorithms. Understanding the trade-offs between different representations and Go-specific optimizations is crucial for efficient implementations.

**Adjacency List Implementation:**

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

**Weighted Graph Representation:**

```go
type Edge struct {
    To     int
    Weight int
}

type WeightedGraph struct {
    nodes map[int][]Edge
}

func NewWeightedGraph() *WeightedGraph {
    return &WeightedGraph{
        nodes: make(map[int][]Edge),
    }
}

func (g *WeightedGraph) AddEdge(from, to, weight int) {
    g.nodes[from] = append(g.nodes[from], Edge{To: to, Weight: weight})
}
```

**Memory-Efficient Representation:**

```go
// For dense graphs or when you know the number of vertices
type DenseGraph struct {
    matrix [][]int
    size   int
}

func NewDenseGraph(size int) *DenseGraph {
    matrix := make([][]int, size)
    for i := range matrix {
        matrix[i] = make([]int, size)
    }
    return &DenseGraph{matrix: matrix, size: size}
}

func (g *DenseGraph) AddEdge(from, to, weight int) {
    g.matrix[from][to] = weight
}
```

**Discussion Questions & Answers:**

**Q1: How do you optimize memory usage for large graphs in Go?**

**Answer:** Memory optimization strategies:

- **Pre-allocate Slices**: Use `make([]int, 0, expectedSize)` to avoid repeated allocations
- **Reuse Data Structures**: Use object pools for frequently created graphs
- **Compressed Representations**: Use bit vectors for boolean adjacency matrices
- **Lazy Initialization**: Only allocate memory when needed
- **Memory Pooling**: Reuse graph structures across multiple operations
- **Garbage Collection**: Minimize allocations to reduce GC pressure

**Q2: What are the performance implications of different Go data structures for graph representation?**

**Answer:** Performance characteristics:

- **Maps vs Slices**: Maps provide O(1) access but have overhead; slices are faster for sequential access
- **Slice Growth**: Dynamic slice growth can cause memory reallocation and copying
- **Memory Layout**: Contiguous memory access is faster than scattered access
- **Cache Locality**: Dense representations have better cache locality
- **Allocation Overhead**: Frequent allocations can impact performance
- **Type Safety**: Go's type system adds minimal runtime overhead

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
