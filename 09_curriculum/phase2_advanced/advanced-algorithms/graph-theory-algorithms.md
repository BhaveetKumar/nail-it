# Graph Theory Algorithms

## Overview

This module covers advanced graph theory algorithms including network flow, matching algorithms, strongly connected components, and graph coloring. These concepts are essential for network optimization, social network analysis, and computational biology.

## Table of Contents

1. [Network Flow Algorithms](#network-flow-algorithms/)
2. [Matching Algorithms](#matching-algorithms/)
3. [Strongly Connected Components](#strongly-connected-components/)
4. [Graph Coloring](#graph-coloring/)
5. [Applications](#applications/)
6. [Complexity Analysis](#complexity-analysis/)
7. [Follow-up Questions](#follow-up-questions/)

## Network Flow Algorithms

### Theory

Network flow algorithms solve optimization problems on flow networks. The maximum flow problem finds the maximum amount of flow that can be sent from a source to a sink through a network.

### Network Flow Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type FlowEdge struct {
    From   int
    To     int
    Capacity int
    Flow   int
}

type FlowNetwork struct {
    Vertices int
    Edges    []FlowEdge
    Adj      [][]int
}

func NewFlowNetwork(vertices int) *FlowNetwork {
    return &FlowNetwork{
        Vertices: vertices,
        Edges:    make([]FlowEdge, 0),
        Adj:      make([][]int, vertices),
    }
}

func (fn *FlowNetwork) AddEdge(from, to, capacity int) {
    edge := FlowEdge{
        From:     from,
        To:       to,
        Capacity: capacity,
        Flow:     0,
    }
    
    fn.Edges = append(fn.Edges, edge)
    fn.Adj[from] = append(fn.Adj[from], len(fn.Edges)-1)
    
    // Add reverse edge
    reverseEdge := FlowEdge{
        From:     to,
        To:       from,
        Capacity: 0,
        Flow:     0,
    }
    
    fn.Edges = append(fn.Edges, reverseEdge)
    fn.Adj[to] = append(fn.Adj[to], len(fn.Edges)-1)
}

func (fn *FlowNetwork) FordFulkerson(source, sink int) int {
    maxFlow := 0
    
    // Create residual graph
    residual := make([]FlowEdge, len(fn.Edges))
    copy(residual, fn.Edges)
    
    // Find augmenting paths
    for {
        path := fn.findAugmentingPath(residual, source, sink)
        if path == nil {
            break
        }
        
        // Find minimum capacity in the path
        minCapacity := math.MaxInt32
        for _, edgeIndex := range path {
            if residual[edgeIndex].Capacity-residual[edgeIndex].Flow < minCapacity {
                minCapacity = residual[edgeIndex].Capacity - residual[edgeIndex].Flow
            }
        }
        
        // Update flow along the path
        for _, edgeIndex := range path {
            residual[edgeIndex].Flow += minCapacity
            // Update reverse edge
            if edgeIndex%2 == 0 {
                residual[edgeIndex+1].Flow -= minCapacity
            } else {
                residual[edgeIndex-1].Flow -= minCapacity
            }
        }
        
        maxFlow += minCapacity
    }
    
    return maxFlow
}

func (fn *FlowNetwork) findAugmentingPath(residual []FlowEdge, source, sink int) []int {
    parent := make([]int, fn.Vertices)
    for i := range parent {
        parent[i] = -1
    }
    
    queue := []int{source}
    parent[source] = source
    
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        
        for _, edgeIndex := range fn.Adj[u] {
            edge := residual[edgeIndex]
            if parent[edge.To] == -1 && edge.Capacity > edge.Flow {
                parent[edge.To] = edgeIndex
                if edge.To == sink {
                    // Reconstruct path
                    path := []int{}
                    v := sink
                    for v != source {
                        path = append([]int{parent[v]}, path...)
                        v = residual[parent[v]].From
                    }
                    return path
                }
                queue = append(queue, edge.To)
            }
        }
    }
    
    return nil
}

func (fn *FlowNetwork) EdmondsKarp(source, sink int) int {
    maxFlow := 0
    
    // Create residual graph
    residual := make([]FlowEdge, len(fn.Edges))
    copy(residual, fn.Edges)
    
    for {
        // Find shortest augmenting path using BFS
        path := fn.findShortestAugmentingPath(residual, source, sink)
        if path == nil {
            break
        }
        
        // Find minimum capacity in the path
        minCapacity := math.MaxInt32
        for _, edgeIndex := range path {
            if residual[edgeIndex].Capacity-residual[edgeIndex].Flow < minCapacity {
                minCapacity = residual[edgeIndex].Capacity - residual[edgeIndex].Flow
            }
        }
        
        // Update flow along the path
        for _, edgeIndex := range path {
            residual[edgeIndex].Flow += minCapacity
            // Update reverse edge
            if edgeIndex%2 == 0 {
                residual[edgeIndex+1].Flow -= minCapacity
            } else {
                residual[edgeIndex-1].Flow -= minCapacity
            }
        }
        
        maxFlow += minCapacity
    }
    
    return maxFlow
}

func (fn *FlowNetwork) findShortestAugmentingPath(residual []FlowEdge, source, sink int) []int {
    parent := make([]int, fn.Vertices)
    for i := range parent {
        parent[i] = -1
    }
    
    queue := []int{source}
    parent[source] = source
    
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        
        for _, edgeIndex := range fn.Adj[u] {
            edge := residual[edgeIndex]
            if parent[edge.To] == -1 && edge.Capacity > edge.Flow {
                parent[edge.To] = edgeIndex
                if edge.To == sink {
                    // Reconstruct path
                    path := []int{}
                    v := sink
                    for v != source {
                        path = append([]int{parent[v]}, path...)
                        v = residual[parent[v]].From
                    }
                    return path
                }
                queue = append(queue, edge.To)
            }
        }
    }
    
    return nil
}

func (fn *FlowNetwork) Dinic(source, sink int) int {
    maxFlow := 0
    
    // Create residual graph
    residual := make([]FlowEdge, len(fn.Edges))
    copy(residual, fn.Edges)
    
    for {
        // Build level graph
        level := fn.buildLevelGraph(residual, source, sink)
        if level[sink] == -1 {
            break
        }
        
        // Find blocking flow
        for {
            flow := fn.findBlockingFlow(residual, source, sink, level)
            if flow == 0 {
                break
            }
            maxFlow += flow
        }
    }
    
    return maxFlow
}

func (fn *FlowNetwork) buildLevelGraph(residual []FlowEdge, source, sink int) []int {
    level := make([]int, fn.Vertices)
    for i := range level {
        level[i] = -1
    }
    
    queue := []int{source}
    level[source] = 0
    
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        
        for _, edgeIndex := range fn.Adj[u] {
            edge := residual[edgeIndex]
            if level[edge.To] == -1 && edge.Capacity > edge.Flow {
                level[edge.To] = level[u] + 1
                if edge.To == sink {
                    return level
                }
                queue = append(queue, edge.To)
            }
        }
    }
    
    return level
}

func (fn *FlowNetwork) findBlockingFlow(residual []FlowEdge, source, sink int, level []int) int {
    if source == sink {
        return math.MaxInt32
    }
    
    flow := 0
    for _, edgeIndex := range fn.Adj[source] {
        edge := residual[edgeIndex]
        if level[edge.To] == level[source]+1 && edge.Capacity > edge.Flow {
            blockingFlow := fn.findBlockingFlow(residual, edge.To, sink, level)
            if blockingFlow > 0 {
                minFlow := blockingFlow
                if edge.Capacity-edge.Flow < minFlow {
                    minFlow = edge.Capacity - edge.Flow
                }
                
                residual[edgeIndex].Flow += minFlow
                if edgeIndex%2 == 0 {
                    residual[edgeIndex+1].Flow -= minFlow
                } else {
                    residual[edgeIndex-1].Flow -= minFlow
                }
                
                flow += minFlow
            }
        }
    }
    
    return flow
}

func (fn *FlowNetwork) MinCut(source, sink int) ([]int, int) {
    // Run Ford-Fulkerson to get residual graph
    fn.FordFulkerson(source, sink)
    
    // Find reachable vertices from source
    visited := make([]bool, fn.Vertices)
    fn.dfs(source, visited)
    
    // Find edges in the min cut
    minCut := []int{}
    for i, edge := range fn.Edges {
        if i%2 == 0 && visited[edge.From] && !visited[edge.To] {
            minCut = append(minCut, i)
        }
    }
    
    return minCut, len(minCut)
}

func (fn *FlowNetwork) dfs(u int, visited []bool) {
    visited[u] = true
    for _, edgeIndex := range fn.Adj[u] {
        edge := fn.Edges[edgeIndex]
        if !visited[edge.To] && edge.Capacity > edge.Flow {
            fn.dfs(edge.To, visited)
        }
    }
}

func main() {
    fn := NewFlowNetwork(6)
    
    fmt.Println("Network Flow Algorithms Demo:")
    
    // Add edges
    fn.AddEdge(0, 1, 16) // source to a
    fn.AddEdge(0, 2, 13) // source to b
    fn.AddEdge(1, 2, 10) // a to b
    fn.AddEdge(1, 3, 12) // a to c
    fn.AddEdge(2, 1, 4)  // b to a
    fn.AddEdge(2, 4, 14) // b to d
    fn.AddEdge(3, 2, 9)  // c to b
    fn.AddEdge(3, 5, 20) // c to sink
    fn.AddEdge(4, 3, 7)  // d to c
    fn.AddEdge(4, 5, 4)  // d to sink
    
    // Ford-Fulkerson
    maxFlow := fn.FordFulkerson(0, 5)
    fmt.Printf("Maximum flow (Ford-Fulkerson): %d\n", maxFlow)
    
    // Edmonds-Karp
    maxFlow2 := fn.EdmondsKarp(0, 5)
    fmt.Printf("Maximum flow (Edmonds-Karp): %d\n", maxFlow2)
    
    // Dinic
    maxFlow3 := fn.Dinic(0, 5)
    fmt.Printf("Maximum flow (Dinic): %d\n", maxFlow3)
    
    // Min cut
    minCut, cutSize := fn.MinCut(0, 5)
    fmt.Printf("Minimum cut size: %d\n", cutSize)
    fmt.Printf("Edges in min cut: %v\n", minCut)
}
```

## Matching Algorithms

### Theory

Matching algorithms find maximum matchings in bipartite graphs. The Hungarian algorithm and Hopcroft-Karp algorithm are efficient methods for solving assignment problems.

### Matching Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type BipartiteGraph struct {
    LeftVertices  int
    RightVertices int
    Edges         [][]int
}

func NewBipartiteGraph(left, right int) *BipartiteGraph {
    return &BipartiteGraph{
        LeftVertices:  left,
        RightVertices: right,
        Edges:         make([][]int, left),
    }
}

func (bg *BipartiteGraph) AddEdge(left, right int) {
    bg.Edges[left] = append(bg.Edges[left], right)
}

func (bg *BipartiteGraph) HopcroftKarp() int {
    // Initialize matching
    pairU := make([]int, bg.LeftVertices)
    pairV := make([]int, bg.RightVertices)
    dist := make([]int, bg.LeftVertices)
    
    for i := range pairU {
        pairU[i] = -1
    }
    for i := range pairV {
        pairV[i] = -1
    }
    
    result := 0
    
    // While there is an augmenting path
    for bg.bfs(pairU, pairV, dist) {
        for u := 0; u < bg.LeftVertices; u++ {
            if pairU[u] == -1 && bg.dfs(u, pairU, pairV, dist) {
                result++
            }
        }
    }
    
    return result
}

func (bg *BipartiteGraph) bfs(pairU, pairV, dist []int) bool {
    queue := []int{}
    
    for u := 0; u < bg.LeftVertices; u++ {
        if pairU[u] == -1 {
            dist[u] = 0
            queue = append(queue, u)
        } else {
            dist[u] = math.MaxInt32
        }
    }
    
    distNil := math.MaxInt32
    
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        
        if dist[u] < distNil {
            for _, v := range bg.Edges[u] {
                if pairV[v] == -1 {
                    distNil = dist[u] + 1
                } else if dist[pairV[v]] == math.MaxInt32 {
                    dist[pairV[v]] = dist[u] + 1
                    queue = append(queue, pairV[v])
                }
            }
        }
    }
    
    return distNil != math.MaxInt32
}

func (bg *BipartiteGraph) dfs(u int, pairU, pairV, dist []int) bool {
    if u != -1 {
        for _, v := range bg.Edges[u] {
            if pairV[v] == -1 || (dist[pairV[v]] == dist[u]+1 && bg.dfs(pairV[v], pairU, pairV, dist)) {
                pairU[u] = v
                pairV[v] = u
                return true
            }
        }
        dist[u] = math.MaxInt32
        return false
    }
    return true
}

func (bg *BipartiteGraph) HungarianAlgorithm(cost [][]int) int {
    n := len(cost)
    if n == 0 {
        return 0
    }
    
    // Initialize
    u := make([]int, n+1)
    v := make([]int, n+1)
    p := make([]int, n+1)
    way := make([]int, n+1)
    
    for i := 1; i <= n; i++ {
        p[0] = i
        j0 := 0
        minv := make([]int, n+1)
        used := make([]bool, n+1)
        
        for k := range minv {
            minv[k] = math.MaxInt32
        }
        
        for {
            used[j0] = true
            i0 := p[j0]
            delta := math.MaxInt32
            j1 := 0
            
            for j := 1; j <= n; j++ {
                if !used[j] {
                    cur := cost[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j] {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if minv[j] < delta {
                        delta = minv[j]
                        j1 = j
                    }
                }
            }
            
            for j := 0; j <= n; j++ {
                if used[j] {
                    u[p[j]] += delta
                    v[j] -= delta
                } else {
                    minv[j] -= delta
                }
            }
            
            j0 = j1
            if p[j0] == 0 {
                break
            }
        }
        
        for j0 != 0 {
            j1 := way[j0]
            p[j0] = p[j1]
            j0 = j1
        }
    }
    
    return -v[0]
}

func (bg *BipartiteGraph) MaximumWeightedMatching(weights [][]int) int {
    n := bg.LeftVertices
    m := bg.RightVertices
    
    if n == 0 || m == 0 {
        return 0
    }
    
    // Initialize
    u := make([]int, n)
    v := make([]int, m)
    p := make([]int, m)
    way := make([]int, m)
    
    for i := 0; i < n; i++ {
        p[0] = i
        j0 := 0
        minv := make([]int, m)
        used := make([]bool, m)
        
        for k := range minv {
            minv[k] = math.MaxInt32
        }
        
        for {
            used[j0] = true
            i0 := p[j0]
            delta := math.MaxInt32
            j1 := 0
            
            for j := 0; j < m; j++ {
                if !used[j] {
                    cur := weights[i0][j] - u[i0] - v[j]
                    if cur < minv[j] {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if minv[j] < delta {
                        delta = minv[j]
                        j1 = j
                    }
                }
            }
            
            for j := 0; j < m; j++ {
                if used[j] {
                    u[p[j]] += delta
                    v[j] -= delta
                } else {
                    minv[j] -= delta
                }
            }
            
            j0 = j1
            if p[j0] == -1 {
                break
            }
        }
        
        for j0 != 0 {
            j1 := way[j0]
            p[j0] = p[j1]
            j0 = j1
        }
    }
    
    return -v[0]
}

func main() {
    // Test bipartite matching
    bg := NewBipartiteGraph(4, 4)
    bg.AddEdge(0, 1)
    bg.AddEdge(0, 2)
    bg.AddEdge(1, 0)
    bg.AddEdge(1, 3)
    bg.AddEdge(2, 2)
    bg.AddEdge(3, 1)
    bg.AddEdge(3, 3)
    
    fmt.Println("Matching Algorithms Demo:")
    
    // Hopcroft-Karp
    maxMatching := bg.HopcroftKarp()
    fmt.Printf("Maximum matching size: %d\n", maxMatching)
    
    // Hungarian algorithm
    cost := [][]int{
        {9, 2, 7, 8},
        {6, 4, 3, 7},
        {5, 8, 1, 8},
        {7, 6, 9, 4},
    }
    
    minCost := bg.HungarianAlgorithm(cost)
    fmt.Printf("Minimum cost assignment: %d\n", minCost)
    
    // Maximum weighted matching
    weights := [][]int{
        {1, 2, 3, 4},
        {2, 4, 6, 8},
        {3, 6, 9, 12},
        {4, 8, 12, 16},
    }
    
    maxWeight := bg.MaximumWeightedMatching(weights)
    fmt.Printf("Maximum weighted matching: %d\n", maxWeight)
}
```

## Strongly Connected Components

### Theory

Strongly connected components (SCCs) are maximal sets of vertices where every vertex is reachable from every other vertex. Tarjan's algorithm and Kosaraju's algorithm are efficient methods for finding SCCs.

### SCC Implementation

#### Golang Implementation

```go
package main

import (
    "fmt"
    "sort"
)

type Graph struct {
    Vertices int
    Adj      [][]int
}

func NewGraph(vertices int) *Graph {
    return &Graph{
        Vertices: vertices,
        Adj:      make([][]int, vertices),
    }
}

func (g *Graph) AddEdge(from, to int) {
    g.Adj[from] = append(g.Adj[from], to)
}

func (g *Graph) TarjanSCC() [][]int {
    index := 0
    stack := []int{}
    onStack := make([]bool, g.Vertices)
    indices := make([]int, g.Vertices)
    lowlinks := make([]int, g.Vertices)
    sccs := [][]int{}
    
    for i := range indices {
        indices[i] = -1
    }
    
    var strongConnect func(int)
    strongConnect = func(v int) {
        indices[v] = index
        lowlinks[v] = index
        index++
        stack = append(stack, v)
        onStack[v] = true
        
        for _, w := range g.Adj[v] {
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
    
    for v := 0; v < g.Vertices; v++ {
        if indices[v] == -1 {
            strongConnect(v)
        }
    }
    
    return sccs
}

func (g *Graph) KosarajuSCC() [][]int {
    // Step 1: Get finish times using DFS
    visited := make([]bool, g.Vertices)
    finishTimes := []int{}
    
    var dfs1 func(int)
    dfs1 = func(v int) {
        visited[v] = true
        for _, w := range g.Adj[v] {
            if !visited[w] {
                dfs1(w)
            }
        }
        finishTimes = append(finishTimes, v)
    }
    
    for v := 0; v < g.Vertices; v++ {
        if !visited[v] {
            dfs1(v)
        }
    }
    
    // Step 2: Reverse the graph
    reversed := g.Reverse()
    
    // Step 3: Process vertices in reverse order of finish times
    visited = make([]bool, g.Vertices)
    sccs := [][]int{}
    
    var dfs2 func(int, []int) []int
    dfs2 = func(v int, scc []int) []int {
        visited[v] = true
        scc = append(scc, v)
        for _, w := range reversed.Adj[v] {
            if !visited[w] {
                scc = dfs2(w, scc)
            }
        }
        return scc
    }
    
    for i := len(finishTimes) - 1; i >= 0; i-- {
        v := finishTimes[i]
        if !visited[v] {
            scc := dfs2(v, []int{})
            sccs = append(sccs, scc)
        }
    }
    
    return sccs
}

func (g *Graph) Reverse() *Graph {
    reversed := NewGraph(g.Vertices)
    for v := 0; v < g.Vertices; v++ {
        for _, w := range g.Adj[v] {
            reversed.AddEdge(w, v)
        }
    }
    return reversed
}

func (g *Graph) CondensationGraph() (*Graph, []int) {
    sccs := g.TarjanSCC()
    n := len(sccs)
    
    // Create mapping from vertex to SCC
    vertexToSCC := make([]int, g.Vertices)
    for i, scc := range sccs {
        for _, v := range scc {
            vertexToSCC[v] = i
        }
    }
    
    // Create condensation graph
    condensed := NewGraph(n)
    edges := make(map[[2]int]bool)
    
    for v := 0; v < g.Vertices; v++ {
        for _, w := range g.Adj[v] {
            sccV := vertexToSCC[v]
            sccW := vertexToSCC[w]
            if sccV != sccW && !edges[[2]int{sccV, sccW}] {
                condensed.AddEdge(sccV, sccW)
                edges[[2]int{sccV, sccW}] = true
            }
        }
    }
    
    return condensed, vertexToSCC
}

func (g *Graph) IsStronglyConnected() bool {
    if g.Vertices == 0 {
        return true
    }
    
    // Check if all vertices are reachable from vertex 0
    visited := make([]bool, g.Vertices)
    g.dfs(0, visited)
    
    for i := 0; i < g.Vertices; i++ {
        if !visited[i] {
            return false
        }
    }
    
    // Check if all vertices can reach vertex 0
    reversed := g.Reverse()
    visited = make([]bool, g.Vertices)
    reversed.dfs(0, visited)
    
    for i := 0; i < g.Vertices; i++ {
        if !visited[i] {
            return false
        }
    }
    
    return true
}

func (g *Graph) dfs(v int, visited []bool) {
    visited[v] = true
    for _, w := range g.Adj[v] {
        if !visited[w] {
            g.dfs(w, visited)
        }
    }
}

func main() {
    g := NewGraph(8)
    
    // Add edges
    g.AddEdge(0, 1)
    g.AddEdge(1, 2)
    g.AddEdge(2, 0)
    g.AddEdge(2, 3)
    g.AddEdge(3, 4)
    g.AddEdge(4, 5)
    g.AddEdge(5, 3)
    g.AddEdge(6, 7)
    
    fmt.Println("Strongly Connected Components Demo:")
    
    // Tarjan's algorithm
    sccs := g.TarjanSCC()
    fmt.Printf("SCCs (Tarjan): %v\n", sccs)
    
    // Kosaraju's algorithm
    sccs2 := g.KosarajuSCC()
    fmt.Printf("SCCs (Kosaraju): %v\n", sccs2)
    
    // Check if strongly connected
    connected := g.IsStronglyConnected()
    fmt.Printf("Graph is strongly connected: %v\n", connected)
    
    // Condensation graph
    condensed, mapping := g.CondensationGraph()
    fmt.Printf("Condensation graph vertices: %d\n", condensed.Vertices)
    fmt.Printf("Vertex to SCC mapping: %v\n", mapping)
}
```

## Follow-up Questions

### 1. Network Flow Algorithms
**Q: What is the difference between Ford-Fulkerson and Edmonds-Karp algorithms?**
A: Ford-Fulkerson uses any augmenting path, while Edmonds-Karp specifically uses the shortest augmenting path (found by BFS), guaranteeing O(VE²) time complexity.

### 2. Matching Algorithms
**Q: When would you use the Hungarian algorithm vs Hopcroft-Karp?**
A: Use Hungarian algorithm for weighted bipartite matching (assignment problems). Use Hopcroft-Karp for unweighted bipartite matching (maximum cardinality matching).

### 3. Strongly Connected Components
**Q: What is the time complexity of Tarjan's algorithm for finding SCCs?**
A: Tarjan's algorithm has O(V + E) time complexity, making it very efficient for finding strongly connected components.

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Ford-Fulkerson | O(E * max_flow) | O(V + E) | Depends on flow value |
| Edmonds-Karp | O(VE²) | O(V + E) | Guaranteed polynomial |
| Dinic | O(V²E) | O(V + E) | Most efficient in practice |
| Hopcroft-Karp | O(E√V) | O(V + E) | Bipartite matching |
| Hungarian | O(V³) | O(V²) | Weighted assignment |
| Tarjan SCC | O(V + E) | O(V) | Very efficient |
| Kosaraju SCC | O(V + E) | O(V + E) | Two DFS passes |

## Applications

1. **Network Flow**: Transportation networks, bipartite matching, image segmentation
2. **Matching Algorithms**: Assignment problems, resource allocation, scheduling
3. **Strongly Connected Components**: Social networks, web page ranking, compiler optimization
4. **Graph Theory**: Network analysis, optimization problems, computational biology

---

**Next**: [Phase 3 Expert](phase3_expert/README.md/) | **Previous**: [Advanced Algorithms](README.md/) | **Up**: [Phase 2 Advanced](README.md/)
