# Advanced Graph Algorithms for Backend Engineers

## Table of Contents
- [Introduction](#introduction/)
- [Advanced Graph Representations](#advanced-graph-representations/)
- [Shortest Path Algorithms](#shortest-path-algorithms/)
- [Minimum Spanning Tree Algorithms](#minimum-spanning-tree-algorithms/)
- [Network Flow Algorithms](#network-flow-algorithms/)
- [Graph Connectivity Algorithms](#graph-connectivity-algorithms/)
- [Advanced Graph Traversal](#advanced-graph-traversal/)
- [Graph Coloring and Matching](#graph-coloring-and-matching/)
- [Dynamic Programming on Graphs](#dynamic-programming-on-graphs/)
- [Parallel Graph Algorithms](#parallel-graph-algorithms/)
- [Real-World Applications](#real-world-applications/)

## Introduction

Advanced graph algorithms are essential for solving complex problems in backend systems, including network routing, social media analysis, recommendation systems, and distributed system optimization. This guide covers sophisticated graph algorithms with practical implementations.

### Why Advanced Graph Algorithms Matter

1. **Network Optimization**: Routing, load balancing, and traffic management
2. **Social Networks**: Friend recommendations, influence analysis
3. **Recommendation Systems**: Collaborative filtering, content recommendation
4. **Distributed Systems**: Consensus algorithms, leader election
5. **Resource Allocation**: Task scheduling, resource optimization

## Advanced Graph Representations

### Adjacency List with Metadata

```go
// Advanced graph representation with metadata
type Graph struct {
    Nodes map[int]*Node
    Edges map[int][]*Edge
    Directed bool
    Weighted bool
}

type Node struct {
    ID       int
    Data     interface{}
    Metadata map[string]interface{}
    InDegree int
    OutDegree int
}

type Edge struct {
    From     int
    To       int
    Weight   float64
    Capacity int
    Flow     int
    Metadata map[string]interface{}
}

func NewGraph(directed, weighted bool) *Graph {
    return &Graph{
        Nodes:    make(map[int]*Node),
        Edges:    make(map[int][]*Edge),
        Directed: directed,
        Weighted: weighted,
    }
}

func (g *Graph) AddNode(id int, data interface{}) {
    g.Nodes[id] = &Node{
        ID:       id,
        Data:     data,
        Metadata: make(map[string]interface{}),
    }
}

func (g *Graph) AddEdge(from, to int, weight float64) {
    edge := &Edge{
        From:     from,
        To:       to,
        Weight:   weight,
        Metadata: make(map[string]interface{}),
    }
    
    g.Edges[from] = append(g.Edges[from], edge)
    g.Nodes[from].OutDegree++
    g.Nodes[to].InDegree++
    
    if !g.Directed {
        reverseEdge := &Edge{
            From:     to,
            To:       from,
            Weight:   weight,
            Metadata: make(map[string]interface{}),
        }
        g.Edges[to] = append(g.Edges[to], reverseEdge)
    }
}
```

### Compressed Sparse Row (CSR) Representation

```go
// CSR representation for memory efficiency
type CSRGraph struct {
    VStart []int    // Start index for each vertex
    EEnd   []int    // End vertices
    EWeight []float64 // Edge weights
    NumVertices int
    NumEdges    int
}

func NewCSRGraph(vertices, edges int) *CSRGraph {
    return &CSRGraph{
        VStart:     make([]int, vertices+1),
        EEnd:       make([]int, edges),
        EWeight:    make([]float64, edges),
        NumVertices: vertices,
        NumEdges:    edges,
    }
}

func (csr *CSRGraph) AddEdge(from, to int, weight float64, edgeIndex int) {
    csr.EEnd[edgeIndex] = to
    csr.EWeight[edgeIndex] = weight
    csr.VStart[from+1]++
}

func (csr *CSRGraph) Build() {
    // Convert counts to start indices
    for i := 1; i <= csr.NumVertices; i++ {
        csr.VStart[i] += csr.VStart[i-1]
    }
}

func (csr *CSRGraph) GetNeighbors(vertex int) []int {
    start := csr.VStart[vertex]
    end := csr.VStart[vertex+1]
    return csr.EEnd[start:end]
}
```

## Shortest Path Algorithms

### Dijkstra's Algorithm with Priority Queue

```go
// Enhanced Dijkstra's algorithm
type DijkstraShortestPath struct {
    Graph     *Graph
    Distances map[int]float64
    Previous  map[int]int
    PQ        *PriorityQueue
}

type PriorityQueue struct {
    items []*PQItem
}

type PQItem struct {
    Vertex   int
    Distance float64
    Index    int
}

func NewDijkstraShortestPath(graph *Graph) *DijkstraShortestPath {
    return &DijkstraShortestPath{
        Graph:     graph,
        Distances: make(map[int]float64),
        Previous:  make(map[int]int),
        PQ:        NewPriorityQueue(),
    }
}

func (dsp *DijkstraShortestPath) FindShortestPath(source int) {
    // Initialize distances
    for nodeID := range dsp.Graph.Nodes {
        dsp.Distances[nodeID] = math.Inf(1)
    }
    dsp.Distances[source] = 0
    
    // Add source to priority queue
    dsp.PQ.Push(&PQItem{Vertex: source, Distance: 0})
    
    for !dsp.PQ.IsEmpty() {
        current := dsp.PQ.Pop()
        
        // Skip if we've already processed this vertex
        if current.Distance > dsp.Distances[current.Vertex] {
            continue
        }
        
        // Relax edges
        for _, edge := range dsp.Graph.Edges[current.Vertex] {
            newDist := dsp.Distances[current.Vertex] + edge.Weight
            
            if newDist < dsp.Distances[edge.To] {
                dsp.Distances[edge.To] = newDist
                dsp.Previous[edge.To] = current.Vertex
                dsp.PQ.Push(&PQItem{Vertex: edge.To, Distance: newDist})
            }
        }
    }
}

func (dsp *DijkstraShortestPath) GetPath(target int) []int {
    if dsp.Distances[target] == math.Inf(1) {
        return nil // No path exists
    }
    
    path := []int{target}
    current := target
    
    for dsp.Previous[current] != 0 {
        current = dsp.Previous[current]
        path = append([]int{current}, path...)
    }
    
    return path
}
```

### A* Algorithm

```go
// A* search algorithm with heuristic
type AStarSearch struct {
    Graph     *Graph
    Heuristic func(int, int) float64
    OpenSet   *PriorityQueue
    CameFrom  map[int]int
    GScore    map[int]float64
    FScore    map[int]float64
}

func NewAStarSearch(graph *Graph, heuristic func(int, int) float64) *AStarSearch {
    return &AStarSearch{
        Graph:     graph,
        Heuristic: heuristic,
        OpenSet:   NewPriorityQueue(),
        CameFrom:  make(map[int]int),
        GScore:    make(map[int]float64),
        FScore:    make(map[int]float64),
    }
}

func (astar *AStarSearch) FindPath(start, goal int) []int {
    // Initialize scores
    for nodeID := range astar.Graph.Nodes {
        astar.GScore[nodeID] = math.Inf(1)
        astar.FScore[nodeID] = math.Inf(1)
    }
    
    astar.GScore[start] = 0
    astar.FScore[start] = astar.Heuristic(start, goal)
    astar.OpenSet.Push(&PQItem{Vertex: start, Distance: astar.FScore[start]})
    
    for !astar.OpenSet.IsEmpty() {
        current := astar.OpenSet.Pop()
        
        if current.Vertex == goal {
            return astar.ReconstructPath(goal)
        }
        
        // Explore neighbors
        for _, edge := range astar.Graph.Edges[current.Vertex] {
            tentativeGScore := astar.GScore[current.Vertex] + edge.Weight
            
            if tentativeGScore < astar.GScore[edge.To] {
                astar.CameFrom[edge.To] = current.Vertex
                astar.GScore[edge.To] = tentativeGScore
                astar.FScore[edge.To] = tentativeGScore + astar.Heuristic(edge.To, goal)
                
                astar.OpenSet.Push(&PQItem{Vertex: edge.To, Distance: astar.FScore[edge.To]})
            }
        }
    }
    
    return nil // No path found
}

func (astar *AStarSearch) ReconstructPath(current int) []int {
    path := []int{current}
    
    for astar.CameFrom[current] != 0 {
        current = astar.CameFrom[current]
        path = append([]int{current}, path...)
    }
    
    return path
}
```

## Minimum Spanning Tree Algorithms

### Kruskal's Algorithm with Union-Find

```go
// Kruskal's MST algorithm with Union-Find
type KruskalMST struct {
    Graph     *Graph
    MST       []*Edge
    UnionFind *UnionFind
}

type UnionFind struct {
    Parent []int
    Rank   []int
}

func NewUnionFind(size int) *UnionFind {
    parent := make([]int, size)
    rank := make([]int, size)
    
    for i := range parent {
        parent[i] = i
    }
    
    return &UnionFind{Parent: parent, Rank: rank}
}

func (uf *UnionFind) Find(x int) int {
    if uf.Parent[x] != x {
        uf.Parent[x] = uf.Find(uf.Parent[x]) // Path compression
    }
    return uf.Parent[x]
}

func (uf *UnionFind) Union(x, y int) bool {
    rootX := uf.Find(x)
    rootY := uf.Find(y)
    
    if rootX == rootY {
        return false // Already connected
    }
    
    // Union by rank
    if uf.Rank[rootX] < uf.Rank[rootY] {
        uf.Parent[rootX] = rootY
    } else if uf.Rank[rootX] > uf.Rank[rootY] {
        uf.Parent[rootY] = rootX
    } else {
        uf.Parent[rootY] = rootX
        uf.Rank[rootX]++
    }
    
    return true
}

func NewKruskalMST(graph *Graph) *KruskalMST {
    return &KruskalMST{
        Graph:     graph,
        MST:       make([]*Edge, 0),
        UnionFind: NewUnionFind(len(graph.Nodes)),
    }
}

func (kruskal *KruskalMST) FindMST() []*Edge {
    // Get all edges and sort by weight
    edges := kruskal.getAllEdges()
    sort.Slice(edges, func(i, j int) bool {
        return edges[i].Weight < edges[j].Weight
    })
    
    // Process edges in order of weight
    for _, edge := range edges {
        if kruskal.UnionFind.Union(edge.From, edge.To) {
            kruskal.MST = append(kruskal.MST, edge)
            
            // Stop when we have n-1 edges
            if len(kruskal.MST) == len(kruskal.Graph.Nodes)-1 {
                break
            }
        }
    }
    
    return kruskal.MST
}

func (kruskal *KruskalMST) getAllEdges() []*Edge {
    edges := make([]*Edge, 0)
    
    for _, edgeList := range kruskal.Graph.Edges {
        edges = append(edges, edgeList...)
    }
    
    return edges
}
```

### Prim's Algorithm with Fibonacci Heap

```go
// Prim's MST algorithm
type PrimMST struct {
    Graph     *Graph
    MST       []*Edge
    InMST     map[int]bool
    Key       map[int]float64
    Parent    map[int]int
    PQ        *PriorityQueue
}

func NewPrimMST(graph *Graph) *PrimMST {
    return &PrimMST{
        Graph:  graph,
        MST:    make([]*Edge, 0),
        InMST:  make(map[int]bool),
        Key:    make(map[int]float64),
        Parent: make(map[int]int),
        PQ:     NewPriorityQueue(),
    }
}

func (prim *PrimMST) FindMST(start int) []*Edge {
    // Initialize keys
    for nodeID := range prim.Graph.Nodes {
        prim.Key[nodeID] = math.Inf(1)
    }
    prim.Key[start] = 0
    prim.PQ.Push(&PQItem{Vertex: start, Distance: 0})
    
    for !prim.PQ.IsEmpty() {
        current := prim.PQ.Pop()
        
        if prim.InMST[current.Vertex] {
            continue
        }
        
        prim.InMST[current.Vertex] = true
        
        // Add edge to MST if not starting vertex
        if prim.Parent[current.Vertex] != 0 {
            edge := &Edge{
                From:   prim.Parent[current.Vertex],
                To:     current.Vertex,
                Weight: prim.Key[current.Vertex],
            }
            prim.MST = append(prim.MST, edge)
        }
        
        // Update keys of neighbors
        for _, edge := range prim.Graph.Edges[current.Vertex] {
            if !prim.InMST[edge.To] && edge.Weight < prim.Key[edge.To] {
                prim.Key[edge.To] = edge.Weight
                prim.Parent[edge.To] = current.Vertex
                prim.PQ.Push(&PQItem{Vertex: edge.To, Distance: edge.Weight})
            }
        }
    }
    
    return prim.MST
}
```

## Network Flow Algorithms

### Ford-Fulkerson with Edmonds-Karp

```go
// Maximum flow using Edmonds-Karp algorithm
type MaxFlow struct {
    Graph     *Graph
    Capacity  map[string]int
    Flow      map[string]int
    Residual  map[string]int
    Source    int
    Sink      int
}

func NewMaxFlow(graph *Graph, source, sink int) *MaxFlow {
    mf := &MaxFlow{
        Graph:    graph,
        Capacity: make(map[string]int),
        Flow:     make(map[string]int),
        Residual: make(map[string]int),
        Source:   source,
        Sink:     sink,
    }
    
    // Initialize capacities and flows
    for from, edges := range graph.Edges {
        for _, edge := range edges {
            key := fmt.Sprintf("%d-%d", from, edge.To)
            mf.Capacity[key] = edge.Capacity
            mf.Flow[key] = 0
            mf.Residual[key] = edge.Capacity
        }
    }
    
    return mf
}

func (mf *MaxFlow) FindMaxFlow() int {
    maxFlow := 0
    
    for {
        // Find augmenting path using BFS
        path := mf.findAugmentingPath()
        if path == nil {
            break
        }
        
        // Find bottleneck capacity
        bottleneck := mf.findBottleneck(path)
        
        // Update flow along the path
        mf.updateFlow(path, bottleneck)
        
        maxFlow += bottleneck
    }
    
    return maxFlow
}

func (mf *MaxFlow) findAugmentingPath() []int {
    queue := []int{mf.Source}
    parent := make(map[int]int)
    visited := make(map[int]bool)
    visited[mf.Source] = true
    
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        
        if current == mf.Sink {
            // Reconstruct path
            path := []int{mf.Sink}
            for current != mf.Source {
                current = parent[current]
                path = append([]int{current}, path...)
            }
            return path
        }
        
        // Check outgoing edges
        for _, edge := range mf.Graph.Edges[current] {
            key := fmt.Sprintf("%d-%d", current, edge.To)
            if !visited[edge.To] && mf.Residual[key] > 0 {
                visited[edge.To] = true
                parent[edge.To] = current
                queue = append(queue, edge.To)
            }
        }
        
        // Check incoming edges (for residual graph)
        for from, edges := range mf.Graph.Edges {
            for _, edge := range edges {
                if edge.To == current {
                    key := fmt.Sprintf("%d-%d", from, current)
                    reverseKey := fmt.Sprintf("%d-%d", current, from)
                    if !visited[from] && mf.Residual[reverseKey] > 0 {
                        visited[from] = true
                        parent[from] = current
                        queue = append(queue, from)
                    }
                }
            }
        }
    }
    
    return nil
}

func (mf *MaxFlow) findBottleneck(path []int) int {
    bottleneck := math.MaxInt32
    
    for i := 0; i < len(path)-1; i++ {
        from := path[i]
        to := path[i+1]
        key := fmt.Sprintf("%d-%d", from, to)
        
        if mf.Residual[key] < bottleneck {
            bottleneck = mf.Residual[key]
        }
    }
    
    return bottleneck
}

func (mf *MaxFlow) updateFlow(path []int, bottleneck int) {
    for i := 0; i < len(path)-1; i++ {
        from := path[i]
        to := path[i+1]
        key := fmt.Sprintf("%d-%d", from, to)
        reverseKey := fmt.Sprintf("%d-%d", to, from)
        
        mf.Flow[key] += bottleneck
        mf.Residual[key] -= bottleneck
        mf.Residual[reverseKey] += bottleneck
    }
}
```

### Min-Cost Max-Flow

```go
// Minimum cost maximum flow using Successive Shortest Path
type MinCostMaxFlow struct {
    Graph     *Graph
    Capacity  map[string]int
    Cost      map[string]float64
    Flow      map[string]int
    Source    int
    Sink      int
}

func NewMinCostMaxFlow(graph *Graph, source, sink int) *MinCostMaxFlow {
    mcmf := &MinCostMaxFlow{
        Graph:    graph,
        Capacity: make(map[string]int),
        Cost:     make(map[string]float64),
        Flow:     make(map[string]int),
        Source:   source,
        Sink:     sink,
    }
    
    // Initialize capacities and costs
    for from, edges := range graph.Edges {
        for _, edge := range edges {
            key := fmt.Sprintf("%d-%d", from, edge.To)
            mcmf.Capacity[key] = edge.Capacity
            mcmf.Cost[key] = edge.Weight
            mcmf.Flow[key] = 0
        }
    }
    
    return mcmf
}

func (mcmf *MinCostMaxFlow) FindMinCostMaxFlow() (int, float64) {
    totalFlow := 0
    totalCost := 0.0
    
    for {
        // Find shortest path using Bellman-Ford
        path, cost := mcmf.findShortestPath()
        if path == nil {
            break
        }
        
        // Find bottleneck capacity
        bottleneck := mcmf.findBottleneck(path)
        
        // Update flow
        mcmf.updateFlow(path, bottleneck)
        
        totalFlow += bottleneck
        totalCost += float64(bottleneck) * cost
    }
    
    return totalFlow, totalCost
}

func (mcmf *MinCostMaxFlow) findShortestPath() ([]int, float64) {
    // Bellman-Ford algorithm for shortest path
    distances := make(map[int]float64)
    parent := make(map[int]int)
    
    for nodeID := range mcmf.Graph.Nodes {
        distances[nodeID] = math.Inf(1)
    }
    distances[mcmf.Source] = 0
    
    // Relax edges V-1 times
    for i := 0; i < len(mcmf.Graph.Nodes)-1; i++ {
        for from, edges := range mcmf.Graph.Edges {
            for _, edge := range edges {
                key := fmt.Sprintf("%d-%d", from, edge.To)
                if mcmf.Flow[key] < mcmf.Capacity[key] {
                    newDist := distances[from] + mcmf.Cost[key]
                    if newDist < distances[edge.To] {
                        distances[edge.To] = newDist
                        parent[edge.To] = from
                    }
                }
            }
        }
    }
    
    if distances[mcmf.Sink] == math.Inf(1) {
        return nil, 0
    }
    
    // Reconstruct path
    path := []int{mcmf.Sink}
    current := mcmf.Sink
    for current != mcmf.Source {
        current = parent[current]
        path = append([]int{current}, path...)
    }
    
    return path, distances[mcmf.Sink]
}
```

## Graph Connectivity Algorithms

### Strongly Connected Components (Tarjan's)

```go
// Tarjan's algorithm for strongly connected components
type TarjanSCC struct {
    Graph     *Graph
    Index     int
    Stack     []int
    OnStack   map[int]bool
    Indices   map[int]int
    LowLinks  map[int]int
    SCCs      [][]int
}

func NewTarjanSCC(graph *Graph) *TarjanSCC {
    return &TarjanSCC{
        Graph:    graph,
        Index:    0,
        Stack:    make([]int, 0),
        OnStack:  make(map[int]bool),
        Indices:  make(map[int]int),
        LowLinks: make(map[int]int),
        SCCs:     make([][]int, 0),
    }
}

func (tarjan *TarjanSCC) FindSCCs() [][]int {
    for nodeID := range tarjan.Graph.Nodes {
        if tarjan.Indices[nodeID] == 0 {
            tarjan.strongConnect(nodeID)
        }
    }
    
    return tarjan.SCCs
}

func (tarjan *TarjanSCC) strongConnect(v int) {
    tarjan.Indices[v] = tarjan.Index
    tarjan.LowLinks[v] = tarjan.Index
    tarjan.Index++
    tarjan.Stack = append(tarjan.Stack, v)
    tarjan.OnStack[v] = true
    
    // Consider successors of v
    for _, edge := range tarjan.Graph.Edges[v] {
        w := edge.To
        if tarjan.Indices[w] == 0 {
            tarjan.strongConnect(w)
            tarjan.LowLinks[v] = min(tarjan.LowLinks[v], tarjan.LowLinks[w])
        } else if tarjan.OnStack[w] {
            tarjan.LowLinks[v] = min(tarjan.LowLinks[v], tarjan.Indices[w])
        }
    }
    
    // If v is a root node, pop the stack and create an SCC
    if tarjan.LowLinks[v] == tarjan.Indices[v] {
        scc := []int{}
        for {
            w := tarjan.Stack[len(tarjan.Stack)-1]
            tarjan.Stack = tarjan.Stack[:len(tarjan.Stack)-1]
            tarjan.OnStack[w] = false
            scc = append(scc, w)
            if w == v {
                break
            }
        }
        tarjan.SCCs = append(tarjan.SCCs, scc)
    }
}
```

### Articulation Points and Bridges

```go
// Find articulation points and bridges
type ArticulationPoints struct {
    Graph        *Graph
    Visited      map[int]bool
    Discovery    map[int]int
    Low          map[int]int
    Parent       map[int]int
    Articulation map[int]bool
    Bridges      []*Edge
    Time         int
}

func NewArticulationPoints(graph *Graph) *ArticulationPoints {
    return &ArticulationPoints{
        Graph:        graph,
        Visited:      make(map[int]bool),
        Discovery:    make(map[int]int),
        Low:          make(map[int]int),
        Parent:       make(map[int]int),
        Articulation: make(map[int]bool),
        Bridges:      make([]*Edge, 0),
        Time:         0,
    }
}

func (ap *ArticulationPoints) FindArticulationPoints() map[int]bool {
    for nodeID := range ap.Graph.Nodes {
        if !ap.Visited[nodeID] {
            ap.dfs(nodeID)
        }
    }
    
    return ap.Articulation
}

func (ap *ArticulationPoints) FindBridges() []*Edge {
    for nodeID := range ap.Graph.Nodes {
        if !ap.Visited[nodeID] {
            ap.dfs(nodeID)
        }
    }
    
    return ap.Bridges
}

func (ap *ArticulationPoints) dfs(u int) {
    ap.Visited[u] = true
    ap.Discovery[u] = ap.Time
    ap.Low[u] = ap.Time
    ap.Time++
    
    children := 0
    
    for _, edge := range ap.Graph.Edges[u] {
        v := edge.To
        if !ap.Visited[v] {
            children++
            ap.Parent[v] = u
            ap.dfs(v)
            
            // Update low value of u
            ap.Low[u] = min(ap.Low[u], ap.Low[v])
            
            // Check for articulation point
            if ap.Parent[u] == -1 && children > 1 {
                ap.Articulation[u] = true
            }
            if ap.Parent[u] != -1 && ap.Low[v] >= ap.Discovery[u] {
                ap.Articulation[u] = true
            }
            
            // Check for bridge
            if ap.Low[v] > ap.Discovery[u] {
                ap.Bridges = append(ap.Bridges, edge)
            }
        } else if v != ap.Parent[u] {
            ap.Low[u] = min(ap.Low[u], ap.Discovery[v])
        }
    }
}
```

## Advanced Graph Traversal

### Bidirectional Search

```go
// Bidirectional BFS for shortest path
type BidirectionalSearch struct {
    Graph     *Graph
    ForwardVisited  map[int]int
    BackwardVisited map[int]int
    ForwardParent   map[int]int
    BackwardParent  map[int]int
}

func NewBidirectionalSearch(graph *Graph) *BidirectionalSearch {
    return &BidirectionalSearch{
        Graph:          graph,
        ForwardVisited:  make(map[int]int),
        BackwardVisited: make(map[int]int),
        ForwardParent:   make(map[int]int),
        BackwardParent:  make(map[int]int),
    }
}

func (bs *BidirectionalSearch) FindPath(start, end int) []int {
    if start == end {
        return []int{start}
    }
    
    forwardQueue := []int{start}
    backwardQueue := []int{end}
    
    bs.ForwardVisited[start] = 0
    bs.BackwardVisited[end] = 0
    
    for len(forwardQueue) > 0 && len(backwardQueue) > 0 {
        // Expand forward search
        if len(forwardQueue) > 0 {
            current := forwardQueue[0]
            forwardQueue = forwardQueue[1:]
            
            for _, edge := range bs.Graph.Edges[current] {
                if _, exists := bs.ForwardVisited[edge.To]; !exists {
                    bs.ForwardVisited[edge.To] = bs.ForwardVisited[current] + 1
                    bs.ForwardParent[edge.To] = current
                    forwardQueue = append(forwardQueue, edge.To)
                    
                    // Check if we've met the backward search
                    if dist, exists := bs.BackwardVisited[edge.To]; exists {
                        return bs.constructPath(edge.To, dist)
                    }
                }
            }
        }
        
        // Expand backward search
        if len(backwardQueue) > 0 {
            current := backwardQueue[0]
            backwardQueue = backwardQueue[1:]
            
            for _, edge := range bs.Graph.Edges[current] {
                if _, exists := bs.BackwardVisited[edge.To]; !exists {
                    bs.BackwardVisited[edge.To] = bs.BackwardVisited[current] + 1
                    bs.BackwardParent[edge.To] = current
                    backwardQueue = append(backwardQueue, edge.To)
                    
                    // Check if we've met the forward search
                    if dist, exists := bs.ForwardVisited[edge.To]; exists {
                        return bs.constructPath(edge.To, dist)
                    }
                }
            }
        }
    }
    
    return nil // No path found
}

func (bs *BidirectionalSearch) constructPath(meetingPoint int, backwardDist int) []int {
    // Construct path from start to meeting point
    forwardPath := []int{meetingPoint}
    current := meetingPoint
    for bs.ForwardParent[current] != 0 {
        current = bs.ForwardParent[current]
        forwardPath = append([]int{current}, forwardPath...)
    }
    
    // Construct path from meeting point to end
    backwardPath := []int{}
    current = meetingPoint
    for bs.BackwardParent[current] != 0 {
        current = bs.BackwardParent[current]
        backwardPath = append(backwardPath, current)
    }
    
    return append(forwardPath, backwardPath...)
}
```

## Graph Coloring and Matching

### Graph Coloring (Greedy)

```go
// Graph coloring using greedy algorithm
type GraphColoring struct {
    Graph     *Graph
    Colors    map[int]int
    Chromatic int
}

func NewGraphColoring(graph *Graph) *GraphColoring {
    return &GraphColoring{
        Graph:  graph,
        Colors: make(map[int]int),
    }
}

func (gc *GraphColoring) ColorGraph() map[int]int {
    // Sort vertices by degree (largest first)
    vertices := make([]int, 0, len(gc.Graph.Nodes))
    for nodeID := range gc.Graph.Nodes {
        vertices = append(vertices, nodeID)
    }
    
    sort.Slice(vertices, func(i, j int) bool {
        return len(gc.Graph.Edges[vertices[i]]) > len(gc.Graph.Edges[vertices[j]])
    })
    
    // Color each vertex
    for _, vertex := range vertices {
        gc.colorVertex(vertex)
    }
    
    // Find chromatic number
    maxColor := 0
    for _, color := range gc.Colors {
        if color > maxColor {
            maxColor = color
        }
    }
    gc.Chromatic = maxColor + 1
    
    return gc.Colors
}

func (gc *GraphColoring) colorVertex(vertex int) {
    // Find the smallest available color
    usedColors := make(map[int]bool)
    
    for _, edge := range gc.Graph.Edges[vertex] {
        if color, exists := gc.Colors[edge.To]; exists {
            usedColors[color] = true
        }
    }
    
    // Assign the smallest unused color
    for color := 0; ; color++ {
        if !usedColors[color] {
            gc.Colors[vertex] = color
            break
        }
    }
}
```

### Maximum Bipartite Matching (Hungarian Algorithm)

```go
// Maximum bipartite matching using Hungarian algorithm
type BipartiteMatching struct {
    Graph     *Graph
    LeftSet   []int
    RightSet  []int
    Matching  map[int]int
    Visited   map[int]bool
}

func NewBipartiteMatching(graph *Graph, leftSet, rightSet []int) *BipartiteMatching {
    return &BipartiteMatching{
        Graph:    graph,
        LeftSet:  leftSet,
        RightSet: rightSet,
        Matching: make(map[int]int),
        Visited:  make(map[int]bool),
    }
}

func (bm *BipartiteMatching) FindMaximumMatching() map[int]int {
    // Try to match each vertex in the left set
    for _, u := range bm.LeftSet {
        // Reset visited array
        for v := range bm.Visited {
            bm.Visited[v] = false
        }
        
        bm.dfs(u)
    }
    
    return bm.Matching
}

func (bm *BipartiteMatching) dfs(u int) bool {
    for _, edge := range bm.Graph.Edges[u] {
        v := edge.To
        if !bm.Visited[v] {
            bm.Visited[v] = true
            
            // If v is not matched or we can find an augmenting path
            if bm.Matching[v] == 0 || bm.dfs(bm.Matching[v]) {
                bm.Matching[u] = v
                bm.Matching[v] = u
                return true
            }
        }
    }
    return false
}
```

## Dynamic Programming on Graphs

### Longest Path in DAG

```go
// Longest path in directed acyclic graph
type LongestPathDAG struct {
    Graph     *Graph
    TopoOrder []int
    Distances map[int]float64
    Parent    map[int]int
}

func NewLongestPathDAG(graph *Graph) *LongestPathDAG {
    return &LongestPathDAG{
        Graph:     graph,
        TopoOrder: make([]int, 0),
        Distances: make(map[int]float64),
        Parent:    make(map[int]int),
    }
}

func (lpd *LongestPathDAG) FindLongestPath(source int) ([]int, float64) {
    // Topological sort
    lpd.topologicalSort()
    
    // Initialize distances
    for nodeID := range lpd.Graph.Nodes {
        lpd.Distances[nodeID] = math.Inf(-1)
    }
    lpd.Distances[source] = 0
    
    // Process vertices in topological order
    for _, u := range lpd.TopoOrder {
        if lpd.Distances[u] != math.Inf(-1) {
            for _, edge := range lpd.Graph.Edges[u] {
                newDist := lpd.Distances[u] + edge.Weight
                if newDist > lpd.Distances[edge.To] {
                    lpd.Distances[edge.To] = newDist
                    lpd.Parent[edge.To] = u
                }
            }
        }
    }
    
    // Find the vertex with maximum distance
    maxDist := math.Inf(-1)
    maxVertex := source
    for vertex, dist := range lpd.Distances {
        if dist > maxDist {
            maxDist = dist
            maxVertex = vertex
        }
    }
    
    // Reconstruct path
    path := lpd.reconstructPath(maxVertex)
    
    return path, maxDist
}

func (lpd *LongestPathDAG) topologicalSort() {
    visited := make(map[int]bool)
    
    var dfs func(int)
    dfs = func(u int) {
        visited[u] = true
        for _, edge := range lpd.Graph.Edges[u] {
            if !visited[edge.To] {
                dfs(edge.To)
            }
        }
        lpd.TopoOrder = append([]int{u}, lpd.TopoOrder...)
    }
    
    for nodeID := range lpd.Graph.Nodes {
        if !visited[nodeID] {
            dfs(nodeID)
        }
    }
}
```

## Parallel Graph Algorithms

### Parallel BFS

```go
// Parallel BFS using goroutines
type ParallelBFS struct {
    Graph     *Graph
    Distances map[int]int
    Visited   map[int]bool
    Mutex     sync.RWMutex
}

func NewParallelBFS(graph *Graph) *ParallelBFS {
    return &ParallelBFS{
        Graph:     graph,
        Distances: make(map[int]int),
        Visited:   make(map[int]bool),
    }
}

func (pbfs *ParallelBFS) FindDistances(source int) map[int]int {
    pbfs.Distances[source] = 0
    pbfs.Visited[source] = true
    
    currentLevel := []int{source}
    level := 0
    
    for len(currentLevel) > 0 {
        level++
        nextLevel := make([]int, 0)
        
        // Process current level in parallel
        var wg sync.WaitGroup
        var mu sync.Mutex
        
        for _, vertex := range currentLevel {
            wg.Add(1)
            go func(v int) {
                defer wg.Done()
                
                neighbors := make([]int, 0)
                pbfs.Mutex.RLock()
                for _, edge := range pbfs.Graph.Edges[v] {
                    neighbors = append(neighbors, edge.To)
                }
                pbfs.Mutex.RUnlock()
                
                for _, neighbor := range neighbors {
                    mu.Lock()
                    if !pbfs.Visited[neighbor] {
                        pbfs.Visited[neighbor] = true
                        pbfs.Distances[neighbor] = level
                        nextLevel = append(nextLevel, neighbor)
                    }
                    mu.Unlock()
                }
            }(vertex)
        }
        
        wg.Wait()
        currentLevel = nextLevel
    }
    
    return pbfs.Distances
}
```

## Real-World Applications

### 1. Social Network Analysis

```go
// Social network analysis using graph algorithms
type SocialNetworkAnalyzer struct {
    Graph *Graph
}

func (sna *SocialNetworkAnalyzer) FindInfluencers() []int {
    // Use PageRank algorithm
    pageRank := make(map[int]float64)
    dampingFactor := 0.85
    iterations := 100
    
    // Initialize PageRank values
    for nodeID := range sna.Graph.Nodes {
        pageRank[nodeID] = 1.0 / float64(len(sna.Graph.Nodes))
    }
    
    // Iterate PageRank
    for i := 0; i < iterations; i++ {
        newPageRank := make(map[int]float64)
        
        for nodeID := range sna.Graph.Nodes {
            newPageRank[nodeID] = (1 - dampingFactor) / float64(len(sna.Graph.Nodes))
            
            for from, edges := range sna.Graph.Edges {
                for _, edge := range edges {
                    if edge.To == nodeID {
                        outDegree := len(sna.Graph.Edges[from])
                        if outDegree > 0 {
                            newPageRank[nodeID] += dampingFactor * pageRank[from] / float64(outDegree)
                        }
                    }
                }
            }
        }
        
        pageRank = newPageRank
    }
    
    // Sort by PageRank
    influencers := make([]int, 0, len(pageRank))
    for nodeID := range pageRank {
        influencers = append(influencers, nodeID)
    }
    
    sort.Slice(influencers, func(i, j int) bool {
        return pageRank[influencers[i]] > pageRank[influencers[j]]
    })
    
    return influencers
}
```

### 2. Recommendation Systems

```go
// Collaborative filtering using graph algorithms
type RecommendationSystem struct {
    UserItemGraph *Graph
    UserSimilarity map[string]float64
}

func (rs *RecommendationSystem) FindRecommendations(userID int, numRecommendations int) []int {
    // Find similar users using Jaccard similarity
    similarUsers := rs.findSimilarUsers(userID)
    
    // Find items liked by similar users
    recommendations := make(map[int]float64)
    
    for _, similarUser := range similarUsers {
        for _, edge := range rs.UserItemGraph.Edges[similarUser] {
            itemID := edge.To
            if !rs.userLikedItem(userID, itemID) {
                recommendations[itemID] += rs.UserSimilarity[fmt.Sprintf("%d-%d", userID, similarUser)]
            }
        }
    }
    
    // Sort by recommendation score
    items := make([]int, 0, len(recommendations))
    for itemID := range recommendations {
        items = append(items, itemID)
    }
    
    sort.Slice(items, func(i, j int) bool {
        return recommendations[items[i]] > recommendations[items[j]]
    })
    
    if len(items) > numRecommendations {
        return items[:numRecommendations]
    }
    return items
}
```

### 3. Network Routing

```go
// Network routing using graph algorithms
type NetworkRouter struct {
    Topology *Graph
    RoutingTable map[int]map[int][]int
}

func (nr *NetworkRouter) BuildRoutingTable() {
    nr.RoutingTable = make(map[int]map[int][]int)
    
    for source := range nr.Topology.Nodes {
        nr.RoutingTable[source] = make(map[int][]int)
        
        // Use Dijkstra's algorithm for each source
        dijkstra := NewDijkstraShortestPath(nr.Topology)
        dijkstra.FindShortestPath(source)
        
        for destination := range nr.Topology.Nodes {
            if source != destination {
                path := dijkstra.GetPath(destination)
                if path != nil {
                    nr.RoutingTable[source][destination] = path
                }
            }
        }
    }
}

func (nr *NetworkRouter) GetRoute(source, destination int) []int {
    if routes, exists := nr.RoutingTable[source]; exists {
        if route, exists := routes[destination]; exists {
            return route
        }
    }
    return nil
}
```

## Conclusion

Advanced graph algorithms are essential tools for backend engineers working on complex systems. These algorithms enable efficient solutions to problems in:

- **Network optimization and routing**
- **Social network analysis and recommendations**
- **Resource allocation and scheduling**
- **Distributed system design**
- **Machine learning and data analysis**

The key to mastering these algorithms is understanding their underlying principles and knowing when to apply each one. Practice implementing these algorithms and applying them to real-world problems to develop deep intuition about graph-based problem solving.

## Additional Resources

- [Introduction to Algorithms - CLRS](https://mitpress.mit.edu/books/introduction-algorithms/)
- [Graph Algorithms - Sedgewick](https://algs4.cs.princeton.edu/40graphs/)
- [NetworkX Documentation](https://networkx.org/documentation/)
- [Graph Theory - Reinhard Diestel](https://www.math.uni-hamburg.de/home/diestel/books/graph.theory/)
- [Competitive Programming - Steven Halim](https://cpbook.net/)
