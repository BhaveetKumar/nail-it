# Graph Algorithms

## Overview

This module covers advanced graph algorithms including shortest path algorithms, minimum spanning trees, topological sorting, strongly connected components, and network flow algorithms. These algorithms are fundamental for solving complex graph problems efficiently.

## Table of Contents

1. [Shortest Path Algorithms](#shortest-path-algorithms)
2. [Minimum Spanning Trees](#minimum-spanning-trees)
3. [Topological Sorting](#topological-sorting)
4. [Strongly Connected Components](#strongly-connected-components)
5. [Network Flow](#network-flow)
6. [Applications](#applications)
7. [Complexity Analysis](#complexity-analysis)
8. [Follow-up Questions](#follow-up-questions)

## Shortest Path Algorithms

### Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights.

#### Theory

- Uses a priority queue to process vertices in order of distance
- Maintains distance array to track shortest distances
- Greedy approach: always processes the closest unvisited vertex

#### Implementations

##### Golang Implementation

```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

type Edge struct {
    to     int
    weight int
}

type Graph struct {
    vertices int
    adjList  [][]Edge
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adjList:  make([][]Edge, vertices),
    }
}

func (g *Graph) AddEdge(from, to, weight int) {
    g.adjList[from] = append(g.adjList[from], Edge{to, weight})
}

type Item struct {
    vertex   int
    distance int
    index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].distance < pq[j].distance
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
    pq[i].index = i
    pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
    n := len(*pq)
    item := x.(*Item)
    item.index = n
    *pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    old[n-1] = nil
    item.index = -1
    *pq = old[0 : n-1]
    return item
}

func (g *Graph) Dijkstra(source int) []int {
    distances := make([]int, g.vertices)
    for i := range distances {
        distances[i] = math.MaxInt32
    }
    distances[source] = 0

    pq := make(PriorityQueue, 0)
    heap.Init(&pq)
    heap.Push(&pq, &Item{vertex: source, distance: 0})

    for pq.Len() > 0 {
        item := heap.Pop(&pq).(*Item)
        u := item.vertex

        if item.distance > distances[u] {
            continue
        }

        for _, edge := range g.adjList[u] {
            v := edge.to
            weight := edge.weight

            if distances[u]+weight < distances[v] {
                distances[v] = distances[u] + weight
                heap.Push(&pq, &Item{vertex: v, distance: distances[v]})
            }
        }
    }

    return distances
}

func main() {
    g := NewGraph(6)
    g.AddEdge(0, 1, 4)
    g.AddEdge(0, 2, 2)
    g.AddEdge(1, 2, 1)
    g.AddEdge(1, 3, 5)
    g.AddEdge(2, 3, 8)
    g.AddEdge(2, 4, 10)
    g.AddEdge(3, 4, 2)
    g.AddEdge(3, 5, 6)
    g.AddEdge(4, 5, 3)

    distances := g.Dijkstra(0)
    fmt.Printf("Shortest distances from vertex 0:\n")
    for i, dist := range distances {
        fmt.Printf("Vertex %d: %d\n", i, dist)
    }
}
```

##### Node.js Implementation

```javascript
class Edge {
    constructor(to, weight) {
        this.to = to;
        this.weight = weight;
    }
}

class Graph {
    constructor(vertices) {
        this.vertices = vertices;
        this.adjList = Array(vertices).fill().map(() => []);
    }

    addEdge(from, to, weight) {
        this.adjList[from].push(new Edge(to, weight));
    }

    dijkstra(source) {
        const distances = Array(this.vertices).fill(Infinity);
        distances[source] = 0;

        const pq = [{ vertex: source, distance: 0 }];

        while (pq.length > 0) {
            pq.sort((a, b) => a.distance - b.distance);
            const { vertex: u, distance: dist } = pq.shift();

            if (dist > distances[u]) continue;

            for (const edge of this.adjList[u]) {
                const v = edge.to;
                const weight = edge.weight;

                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    pq.push({ vertex: v, distance: distances[v] });
                }
            }
        }

        return distances;
    }
}

// Example usage
const g = new Graph(6);
g.addEdge(0, 1, 4);
g.addEdge(0, 2, 2);
g.addEdge(1, 2, 1);
g.addEdge(1, 3, 5);
g.addEdge(2, 3, 8);
g.addEdge(2, 4, 10);
g.addEdge(3, 4, 2);
g.addEdge(3, 5, 6);
g.addEdge(4, 5, 3);

const distances = g.dijkstra(0);
console.log('Shortest distances from vertex 0:');
distances.forEach((dist, i) => {
    console.log(`Vertex ${i}: ${dist}`);
});
```

### Bellman-Ford Algorithm

Bellman-Ford algorithm finds shortest paths from a source vertex to all other vertices, even with negative edge weights (but no negative cycles).

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type Edge struct {
    from   int
    to     int
    weight int
}

func BellmanFord(edges []Edge, vertices, source int) ([]int, bool) {
    distances := make([]int, vertices)
    for i := range distances {
        distances[i] = math.MaxInt32
    }
    distances[source] = 0

    // Relax edges V-1 times
    for i := 0; i < vertices-1; i++ {
        for _, edge := range edges {
            if distances[edge.from] != math.MaxInt32 && 
               distances[edge.from]+edge.weight < distances[edge.to] {
                distances[edge.to] = distances[edge.from] + edge.weight
            }
        }
    }

    // Check for negative cycles
    for _, edge := range edges {
        if distances[edge.from] != math.MaxInt32 && 
           distances[edge.from]+edge.weight < distances[edge.to] {
            return nil, false // Negative cycle detected
        }
    }

    return distances, true
}

func main() {
    edges := []Edge{
        {0, 1, 4}, {0, 2, 2}, {1, 2, 1}, {1, 3, 5},
        {2, 3, 8}, {2, 4, 10}, {3, 4, 2}, {3, 5, 6},
        {4, 5, 3},
    }

    distances, success := BellmanFord(edges, 6, 0)
    if success {
        fmt.Printf("Shortest distances from vertex 0:\n")
        for i, dist := range distances {
            fmt.Printf("Vertex %d: %d\n", i, dist)
        }
    } else {
        fmt.Println("Negative cycle detected!")
    }
}
```

## Minimum Spanning Trees

### Kruskal's Algorithm

Kruskal's algorithm finds the minimum spanning tree of a connected, undirected graph using a union-find data structure.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type Edge struct {
    from   int
    to     int
    weight int
}

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
    return &UnionFind{parent: parent, rank: rank}
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x])
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) bool {
    px, py := uf.Find(x), uf.Find(y)
    if px == py {
        return false
    }

    if uf.rank[px] < uf.rank[py] {
        uf.parent[px] = py
    } else if uf.rank[px] > uf.rank[py] {
        uf.parent[py] = px
    } else {
        uf.parent[py] = px
        uf.rank[px]++
    }
    return true
}

func Kruskal(edges []Edge, vertices int) []Edge {
    sort.Slice(edges, func(i, j int) bool {
        return edges[i].weight < edges[j].weight
    })

    uf := NewUnionFind(vertices)
    mst := []Edge{}

    for _, edge := range edges {
        if uf.Union(edge.from, edge.to) {
            mst = append(mst, edge)
            if len(mst) == vertices-1 {
                break
            }
        }
    }

    return mst
}

func main() {
    edges := []Edge{
        {0, 1, 4}, {0, 2, 2}, {1, 2, 1}, {1, 3, 5},
        {2, 3, 8}, {2, 4, 10}, {3, 4, 2}, {3, 5, 6},
        {4, 5, 3},
    }

    mst := Kruskal(edges, 6)
    totalWeight := 0

    fmt.Println("Minimum Spanning Tree edges:")
    for _, edge := range mst {
        fmt.Printf("Edge %d-%d: weight %d\n", edge.from, edge.to, edge.weight)
        totalWeight += edge.weight
    }
    fmt.Printf("Total weight: %d\n", totalWeight)
}
```

### Prim's Algorithm

Prim's algorithm finds the minimum spanning tree by growing it one vertex at a time.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

type Edge struct {
    to     int
    weight int
}

type Graph struct {
    vertices int
    adjList  [][]Edge
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adjList:  make([][]Edge, vertices),
    }
}

func (g *Graph) AddEdge(from, to, weight int) {
    g.adjList[from] = append(g.adjList[from], Edge{to, weight})
    g.adjList[to] = append(g.adjList[to], Edge{from, weight})
}

type Item struct {
    vertex   int
    weight   int
    index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].weight < pq[j].weight
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
    pq[i].index = i
    pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
    n := len(*pq)
    item := x.(*Item)
    item.index = n
    *pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    old[n-1] = nil
    item.index = -1
    *pq = old[0 : n-1]
    return item
}

func (g *Graph) Prim() []Edge {
    mst := []Edge{}
    visited := make([]bool, g.vertices)
    key := make([]int, g.vertices)
    parent := make([]int, g.vertices)

    for i := range key {
        key[i] = math.MaxInt32
    }

    key[0] = 0
    parent[0] = -1

    pq := make(PriorityQueue, 0)
    heap.Init(&pq)
    heap.Push(&pq, &Item{vertex: 0, weight: 0})

    for pq.Len() > 0 {
        item := heap.Pop(&pq).(*Item)
        u := item.vertex

        if visited[u] {
            continue
        }

        visited[u] = true

        if parent[u] != -1 {
            mst = append(mst, Edge{from: parent[u], to: u, weight: key[u]})
        }

        for _, edge := range g.adjList[u] {
            v := edge.to
            weight := edge.weight

            if !visited[v] && weight < key[v] {
                key[v] = weight
                parent[v] = u
                heap.Push(&pq, &Item{vertex: v, weight: weight})
            }
        }
    }

    return mst
}

func main() {
    g := NewGraph(6)
    g.AddEdge(0, 1, 4)
    g.AddEdge(0, 2, 2)
    g.AddEdge(1, 2, 1)
    g.AddEdge(1, 3, 5)
    g.AddEdge(2, 3, 8)
    g.AddEdge(2, 4, 10)
    g.AddEdge(3, 4, 2)
    g.AddEdge(3, 5, 6)
    g.AddEdge(4, 5, 3)

    mst := g.Prim()
    totalWeight := 0

    fmt.Println("Minimum Spanning Tree edges:")
    for _, edge := range mst {
        fmt.Printf("Edge %d-%d: weight %d\n", edge.from, edge.to, edge.weight)
        totalWeight += edge.weight
    }
    fmt.Printf("Total weight: %d\n", totalWeight)
}
```

## Topological Sorting

### Kahn's Algorithm

Kahn's algorithm finds a topological ordering of vertices in a directed acyclic graph (DAG).

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
)

type Graph struct {
    vertices int
    adjList  [][]int
    inDegree []int
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adjList:  make([][]int, vertices),
        inDegree: make([]int, vertices),
    }
}

func (g *Graph) AddEdge(from, to int) {
    g.adjList[from] = append(g.adjList[from], to)
    g.inDegree[to]++
}

func (g *Graph) TopologicalSort() ([]int, bool) {
    queue := []int{}
    result := []int{}

    // Find all vertices with in-degree 0
    for i := 0; i < g.vertices; i++ {
        if g.inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }

    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        result = append(result, u)

        for _, v := range g.adjList[u] {
            g.inDegree[v]--
            if g.inDegree[v] == 0 {
                queue = append(queue, v)
            }
        }
    }

    // Check if all vertices are processed
    if len(result) != g.vertices {
        return nil, false // Cycle detected
    }

    return result, true
}

func main() {
    g := NewGraph(6)
    g.AddEdge(5, 2)
    g.AddEdge(5, 0)
    g.AddEdge(4, 0)
    g.AddEdge(4, 1)
    g.AddEdge(2, 3)
    g.AddEdge(3, 1)

    result, success := g.TopologicalSort()
    if success {
        fmt.Println("Topological ordering:")
        for _, vertex := range result {
            fmt.Printf("%d ", vertex)
        }
        fmt.Println()
    } else {
        fmt.Println("Cycle detected - no topological ordering possible")
    }
}
```

## Strongly Connected Components

### Tarjan's Algorithm

Tarjan's algorithm finds all strongly connected components in a directed graph.

#### Implementations

##### Golang Implementation

```go
package main

import (
    "fmt"
)

type Graph struct {
    vertices int
    adjList  [][]int
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        vertices: vertices,
        adjList:  make([][]int, vertices),
    }
}

func (g *Graph) AddEdge(from, to int) {
    g.adjList[from] = append(g.adjList[from], to)
}

func (g *Graph) TarjanSCC() [][]int {
    index := 0
    stack := []int{}
    onStack := make([]bool, g.vertices)
    indices := make([]int, g.vertices)
    lowlinks := make([]int, g.vertices)
    sccs := [][]int{}

    for i := 0; i < g.vertices; i++ {
        indices[i] = -1
    }

    var strongConnect func(int)
    strongConnect = func(v int) {
        indices[v] = index
        lowlinks[v] = index
        index++
        stack = append(stack, v)
        onStack[v] = true

        for _, w := range g.adjList[v] {
            if indices[w] == -1 {
                strongConnect(w)
                if lowlinks[w] < lowlinks[v] {
                    lowlinks[v] = lowlinks[w]
                }
            } else if onStack[w] {
                if indices[w] < lowlinks[v] {
                    lowlinks[v] = indices[w]
                }
            }
        }

        if lowlinks[v] == indices[v] {
            scc := []int{}
            for {
                w := stack[len(stack)-1]
                stack = stack[:len(stack)-1]
                onStack[w] = false
                scc = append(scc, w)
                if w == v {
                    break
                }
            }
            sccs = append(sccs, scc)
        }
    }

    for v := 0; v < g.vertices; v++ {
        if indices[v] == -1 {
            strongConnect(v)
        }
    }

    return sccs
}

func main() {
    g := NewGraph(8)
    g.AddEdge(0, 1)
    g.AddEdge(1, 2)
    g.AddEdge(2, 0)
    g.AddEdge(2, 3)
    g.AddEdge(3, 4)
    g.AddEdge(4, 5)
    g.AddEdge(5, 6)
    g.AddEdge(6, 4)
    g.AddEdge(6, 7)

    sccs := g.TarjanSCC()
    fmt.Println("Strongly Connected Components:")
    for i, scc := range sccs {
        fmt.Printf("SCC %d: %v\n", i+1, scc)
    }
}
```

## Follow-up Questions

### 1. Algorithm Selection
**Q: When would you use Dijkstra's algorithm vs Bellman-Ford?**
A: Use Dijkstra for graphs with non-negative edge weights as it's more efficient (O((V+E)log V) vs O(VE)). Use Bellman-Ford when you have negative edge weights or need to detect negative cycles.

### 2. MST Algorithms
**Q: What's the difference between Kruskal's and Prim's algorithms?**
A: Kruskal's processes edges in sorted order and uses union-find, making it better for sparse graphs. Prim's grows the MST from a starting vertex using a priority queue, making it better for dense graphs.

### 3. Graph Representation
**Q: How do you choose between adjacency list and adjacency matrix?**
A: Use adjacency list for sparse graphs (space efficient, O(V+E) space). Use adjacency matrix for dense graphs or when you need O(1) edge lookup (O(V²) space).

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Dijkstra | O((V+E)log V) | O(V) | With binary heap |
| Bellman-Ford | O(VE) | O(V) | Handles negative weights |
| Kruskal | O(E log E) | O(V) | With union-find |
| Prim | O((V+E)log V) | O(V) | With binary heap |
| Topological Sort | O(V+E) | O(V) | Kahn's algorithm |
| Tarjan's SCC | O(V+E) | O(V) | DFS-based |

## Applications

1. **Shortest Path**: GPS navigation, network routing
2. **MST**: Network design, clustering
3. **Topological Sort**: Task scheduling, dependency resolution
4. **SCC**: Social network analysis, compiler optimization
5. **Network Flow**: Maximum flow, bipartite matching

---

**Next**: [Dynamic Programming Advanced](dynamic-programming-advanced.md) | **Previous**: [Advanced DSA](README.md) | **Up**: [Advanced DSA](README.md)


## Network Flow

<!-- AUTO-GENERATED ANCHOR: originally referenced as #network-flow -->

Placeholder content. Please replace with proper section.
