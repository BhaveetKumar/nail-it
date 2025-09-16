# Advanced Algorithms

## Table of Contents

1. [Overview](#overview/)
2. [Advanced Graph Algorithms](#advanced-graph-algorithms/)
3. [String Algorithms](#string-algorithms/)
4. [Computational Geometry](#computational-geometry/)
5. [Number Theory](#number-theory/)
6. [Combinatorics](#combinatorics/)
7. [Game Theory](#game-theory/)
8. [Optimization Algorithms](#optimization-algorithms/)
9. [Implementations](#implementations/)
10. [Follow-up Questions](#follow-up-questions/)
11. [Sources](#sources/)
12. [Projects](#projects/)

## Overview

### Learning Objectives

- Master advanced graph algorithms and network flow
- Implement sophisticated string algorithms
- Solve computational geometry problems
- Apply number theory and combinatorics
- Design game theory strategies
- Optimize algorithms for performance

### What are Advanced Algorithms?

Advanced Algorithms cover sophisticated algorithmic techniques used to solve complex computational problems efficiently, including graph theory, string processing, geometry, and optimization.

## Advanced Graph Algorithms

### 1. Network Flow Algorithms

#### Max Flow - Ford-Fulkerson Algorithm
```go
package main

import (
    "fmt"
    "math"
)

type Edge struct {
    to     int
    cap    int
    flow   int
    rev    int
}

type MaxFlow struct {
    graph [][]Edge
    n     int
}

func NewMaxFlow(n int) *MaxFlow {
    return &MaxFlow{
        graph: make([][]Edge, n),
        n:     n,
    }
}

func (mf *MaxFlow) AddEdge(from, to, cap int) {
    forward := Edge{to: to, cap: cap, flow: 0, rev: len(mf.graph[to])}
    backward := Edge{to: from, cap: 0, flow: 0, rev: len(mf.graph[from])}
    
    mf.graph[from] = append(mf.graph[from], forward)
    mf.graph[to] = append(mf.graph[to], backward)
}

func (mf *MaxFlow) MaxFlow(source, sink int) int {
    totalFlow := 0
    parent := make([]int, mf.n)
    
    for {
        // BFS to find augmenting path
        for i := range parent {
            parent[i] = -1
        }
        
        queue := []int{source}
        parent[source] = source
        
        for len(queue) > 0 {
            u := queue[0]
            queue = queue[1:]
            
            if u == sink {
                break
            }
            
            for i, edge := range mf.graph[u] {
                if parent[edge.to] == -1 && edge.cap > edge.flow {
                    parent[edge.to] = u
                    queue = append(queue, edge.to)
                }
            }
        }
        
        if parent[sink] == -1 {
            break // No augmenting path found
        }
        
        // Find minimum residual capacity
        pathFlow := math.MaxInt32
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            for i, edge := range mf.graph[u] {
                if edge.to == v {
                    pathFlow = min(pathFlow, edge.cap-edge.flow)
                    break
                }
            }
        }
        
        // Update flow along the path
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            for i := range mf.graph[u] {
                if mf.graph[u][i].to == v {
                    mf.graph[u][i].flow += pathFlow
                    mf.graph[v][mf.graph[u][i].rev].flow -= pathFlow
                    break
                }
            }
        }
        
        totalFlow += pathFlow
    }
    
    return totalFlow
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    // Create a flow network
    mf := NewMaxFlow(6)
    
    // Add edges (from, to, capacity)
    mf.AddEdge(0, 1, 16) // source to node 1
    mf.AddEdge(0, 2, 13) // source to node 2
    mf.AddEdge(1, 2, 10) // node 1 to node 2
    mf.AddEdge(1, 3, 12) // node 1 to node 3
    mf.AddEdge(2, 1, 4)  // node 2 to node 1
    mf.AddEdge(2, 4, 14) // node 2 to node 4
    mf.AddEdge(3, 2, 9)  // node 3 to node 2
    mf.AddEdge(3, 5, 20) // node 3 to sink
    mf.AddEdge(4, 3, 7)  // node 4 to node 3
    mf.AddEdge(4, 5, 4)  // node 4 to sink
    
    maxFlow := mf.MaxFlow(0, 5)
    fmt.Printf("Maximum flow: %d\n", maxFlow)
}
```

#### Min Cost Max Flow
```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

type Edge struct {
    to     int
    cap    int
    cost   int
    flow   int
    rev    int
}

type MinCostMaxFlow struct {
    graph [][]Edge
    n     int
}

func NewMinCostMaxFlow(n int) *MinCostMaxFlow {
    return &MinCostMaxFlow{
        graph: make([][]Edge, n),
        n:     n,
    }
}

func (mcmf *MinCostMaxFlow) AddEdge(from, to, cap, cost int) {
    forward := Edge{to: to, cap: cap, cost: cost, flow: 0, rev: len(mcmf.graph[to])}
    backward := Edge{to: from, cap: 0, cost: -cost, flow: 0, rev: len(mcmf.graph[from])}
    
    mcmf.graph[from] = append(mcmf.graph[from], forward)
    mcmf.graph[to] = append(mcmf.graph[to], backward)
}

type State struct {
    cost int
    node int
}

type PriorityQueue []State

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].cost < pq[j].cost }
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(State))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}

func (mcmf *MinCostMaxFlow) MinCostMaxFlow(source, sink int) (int, int) {
    totalFlow := 0
    totalCost := 0
    
    for {
        // Bellman-Ford to find shortest path
        dist := make([]int, mcmf.n)
        parent := make([]int, mcmf.n)
        parentEdge := make([]int, mcmf.n)
        
        for i := range dist {
            dist[i] = math.MaxInt32
            parent[i] = -1
        }
        dist[source] = 0
        
        // Relax edges
        for i := 0; i < mcmf.n-1; i++ {
            for u := 0; u < mcmf.n; u++ {
                for j, edge := range mcmf.graph[u] {
                    if edge.cap > edge.flow && dist[u] != math.MaxInt32 {
                        if dist[u]+edge.cost < dist[edge.to] {
                            dist[edge.to] = dist[u] + edge.cost
                            parent[edge.to] = u
                            parentEdge[edge.to] = j
                        }
                    }
                }
            }
        }
        
        if parent[sink] == -1 {
            break // No augmenting path found
        }
        
        // Find minimum residual capacity
        pathFlow := math.MaxInt32
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            edge := mcmf.graph[u][parentEdge[v]]
            pathFlow = min(pathFlow, edge.cap-edge.flow)
        }
        
        // Update flow along the path
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            edge := &mcmf.graph[u][parentEdge[v]]
            edge.flow += pathFlow
            mcmf.graph[v][edge.rev].flow -= pathFlow
            totalCost += pathFlow * edge.cost
        }
        
        totalFlow += pathFlow
    }
    
    return totalFlow, totalCost
}

func main() {
    mcmf := NewMinCostMaxFlow(4)
    
    // Add edges (from, to, capacity, cost)
    mcmf.AddEdge(0, 1, 10, 4) // source to node 1
    mcmf.AddEdge(0, 2, 5, 1)  // source to node 2
    mcmf.AddEdge(1, 2, 8, 2)  // node 1 to node 2
    mcmf.AddEdge(1, 3, 5, 6)  // node 1 to sink
    mcmf.AddEdge(2, 3, 10, 3) // node 2 to sink
    
    flow, cost := mcmf.MinCostMaxFlow(0, 3)
    fmt.Printf("Maximum flow: %d, Minimum cost: %d\n", flow, cost)
}
```

### 2. Advanced Graph Traversal

#### A* Search Algorithm
```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

type Node struct {
    x, y int
    g, h, f float64
    parent *Node
}

type PriorityQueue []*Node

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].f < pq[j].f }
func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(*Node))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}

type AStar struct {
    grid     [][]int
    width    int
    height   int
    start    *Node
    goal     *Node
    openSet  PriorityQueue
    closedSet map[string]bool
}

func NewAStar(grid [][]int) *AStar {
    return &AStar{
        grid:      grid,
        width:     len(grid[0]),
        height:    len(grid),
        closedSet: make(map[string]bool),
    }
}

func (as *AStar) heuristic(a, b *Node) float64 {
    return math.Sqrt(float64((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)))
}

func (as *AStar) getKey(node *Node) string {
    return fmt.Sprintf("%d,%d", node.x, node.y)
}

func (as *AStar) isValid(x, y int) bool {
    return x >= 0 && x < as.width && y >= 0 && y < as.height && as.grid[y][x] == 0
}

func (as *AStar) getNeighbors(node *Node) []*Node {
    neighbors := []*Node{}
    directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
    
    for _, dir := range directions {
        newX, newY := node.x+dir[0], node.y+dir[1]
        if as.isValid(newX, newY) {
            neighbors = append(neighbors, &Node{x: newX, y: newY})
        }
    }
    
    return neighbors
}

func (as *AStar) FindPath(startX, startY, goalX, goalY int) []*Node {
    as.start = &Node{x: startX, y: startY, g: 0}
    as.goal = &Node{x: goalX, y: goalY}
    
    as.start.h = as.heuristic(as.start, as.goal)
    as.start.f = as.start.g + as.start.h
    
    as.openSet = PriorityQueue{as.start}
    heap.Init(&as.openSet)
    
    for as.openSet.Len() > 0 {
        current := heap.Pop(&as.openSet).(*Node)
        
        if current.x == as.goal.x && current.y == as.goal.y {
            return as.reconstructPath(current)
        }
        
        as.closedSet[as.getKey(current)] = true
        
        for _, neighbor := range as.getNeighbors(current) {
            if as.closedSet[as.getKey(neighbor)] {
                continue
            }
            
            tentativeG := current.g + as.heuristic(current, neighbor)
            
            // Check if this path to neighbor is better
            neighbor.g = tentativeG
            neighbor.h = as.heuristic(neighbor, as.goal)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current
            
            // Add to open set if not already there
            if !as.inOpenSet(neighbor) {
                heap.Push(&as.openSet, neighbor)
            }
        }
    }
    
    return nil // No path found
}

func (as *AStar) inOpenSet(node *Node) bool {
    for _, n := range as.openSet {
        if n.x == node.x && n.y == node.y {
            return true
        }
    }
    return false
}

func (as *AStar) reconstructPath(node *Node) []*Node {
    path := []*Node{}
    current := node
    
    for current != nil {
        path = append([]*Node{current}, path...)
        current = current.parent
    }
    
    return path
}

func main() {
    // 0 = walkable, 1 = obstacle
    grid := [][]int{
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 0, 0},
        {0, 0, 1, 0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    }
    
    astar := NewAStar(grid)
    path := astar.FindPath(0, 0, 7, 6)
    
    if path != nil {
        fmt.Println("Path found:")
        for _, node := range path {
            fmt.Printf("(%d, %d) ", node.x, node.y)
        }
        fmt.Println()
    } else {
        fmt.Println("No path found")
    }
}
```

## String Algorithms

### 1. Advanced String Matching

#### KMP Algorithm
```go
package main

import "fmt"

func buildLPS(pattern string) []int {
    m := len(pattern)
    lps := make([]int, m)
    length := 0
    i := 1
    
    for i < m {
        if pattern[i] == pattern[length] {
            length++
            lps[i] = length
            i++
        } else {
            if length != 0 {
                length = lps[length-1]
            } else {
                lps[i] = 0
                i++
            }
        }
    }
    
    return lps
}

func KMP(text, pattern string) []int {
    n := len(text)
    m := len(pattern)
    lps := buildLPS(pattern)
    
    var matches []int
    i, j := 0, 0
    
    for i < n {
        if pattern[j] == text[i] {
            i++
            j++
        }
        
        if j == m {
            matches = append(matches, i-j)
            j = lps[j-1]
        } else if i < n && pattern[j] != text[i] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    
    return matches
}

func main() {
    text := "ABABDABACDABABCABAB"
    pattern := "ABABCABAB"
    
    matches := KMP(text, pattern)
    fmt.Printf("Pattern found at positions: %v\n", matches)
}
```

#### Rabin-Karp Algorithm
```go
package main

import "fmt"

const base = 256
const mod = 101

func rabinKarp(text, pattern string) []int {
    n := len(text)
    m := len(pattern)
    
    if m > n {
        return []int{}
    }
    
    // Calculate hash of pattern and first window of text
    patternHash := 0
    textHash := 0
    h := 1
    
    // Calculate h = pow(base, m-1) % mod
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    // Calculate hash of pattern and first window
    for i := 0; i < m; i++ {
        patternHash = (base*patternHash + int(pattern[i])) % mod
        textHash = (base*textHash + int(text[i])) % mod
    }
    
    var matches []int
    
    // Slide the pattern over text
    for i := 0; i <= n-m; i++ {
        // Check if hash values match
        if patternHash == textHash {
            // Check characters one by one
            j := 0
            for j < m && text[i+j] == pattern[j] {
                j++
            }
            
            if j == m {
                matches = append(matches, i)
            }
        }
        
        // Calculate hash for next window
        if i < n-m {
            textHash = (base*(textHash-int(text[i])*h) + int(text[i+m])) % mod
            
            // Handle negative values
            if textHash < 0 {
                textHash += mod
            }
        }
    }
    
    return matches
}

func main() {
    text := "GEEKS FOR GEEKS"
    pattern := "GEEK"
    
    matches := rabinKarp(text, pattern)
    fmt.Printf("Pattern found at positions: %v\n", matches)
}
```

### 2. Suffix Arrays and Trees

#### Suffix Array Construction
```go
package main

import (
    "fmt"
    "sort"
)

type Suffix struct {
    index int
    rank  [2]int
}

func buildSuffixArray(text string) []int {
    n := len(text)
    suffixes := make([]Suffix, n)
    
    // Initialize suffixes with single character ranks
    for i := 0; i < n; i++ {
        suffixes[i].index = i
        suffixes[i].rank[0] = int(text[i])
        if i+1 < n {
            suffixes[i].rank[1] = int(text[i+1])
        } else {
            suffixes[i].rank[1] = -1
        }
    }
    
    // Sort suffixes by rank
    sort.Slice(suffixes, func(i, j int) bool {
        if suffixes[i].rank[0] != suffixes[j].rank[0] {
            return suffixes[i].rank[0] < suffixes[j].rank[0]
        }
        return suffixes[i].rank[1] < suffixes[j].rank[1]
    })
    
    // Process suffixes with length 4, 8, 16, ...
    for k := 4; k < 2*n; k *= 2 {
        // Assign new ranks
        rank := 0
        prevRank := suffixes[0].rank[0]
        suffixes[0].rank[0] = rank
        
        for i := 1; i < n; i++ {
            if suffixes[i].rank[0] == prevRank && suffixes[i].rank[1] == suffixes[i-1].rank[1] {
                prevRank = suffixes[i].rank[0]
                suffixes[i].rank[0] = rank
            } else {
                prevRank = suffixes[i].rank[0]
                rank++
                suffixes[i].rank[0] = rank
            }
        }
        
        // Assign next rank
        for i := 0; i < n; i++ {
            nextIndex := suffixes[i].index + k/2
            if nextIndex < n {
                suffixes[i].rank[1] = suffixes[getSuffixIndex(suffixes, nextIndex)].rank[0]
            } else {
                suffixes[i].rank[1] = -1
            }
        }
        
        // Sort suffixes by new ranks
        sort.Slice(suffixes, func(i, j int) bool {
            if suffixes[i].rank[0] != suffixes[j].rank[0] {
                return suffixes[i].rank[0] < suffixes[j].rank[0]
            }
            return suffixes[i].rank[1] < suffixes[j].rank[1]
        })
    }
    
    // Extract suffix array
    suffixArray := make([]int, n)
    for i := 0; i < n; i++ {
        suffixArray[i] = suffixes[i].index
    }
    
    return suffixArray
}

func getSuffixIndex(suffixes []Suffix, index int) int {
    for i, suffix := range suffixes {
        if suffix.index == index {
            return i
        }
    }
    return -1
}

func main() {
    text := "banana"
    suffixArray := buildSuffixArray(text)
    
    fmt.Printf("Suffix Array for '%s':\n", text)
    for i, index := range suffixArray {
        fmt.Printf("%d: %s\n", i, text[index:])
    }
}
```

## Computational Geometry

### 1. Convex Hull

#### Graham Scan Algorithm
```go
package main

import (
    "fmt"
    "math"
    "sort"
)

type Point struct {
    x, y float64
}

func (p Point) String() string {
    return fmt.Sprintf("(%.2f, %.2f)", p.x, p.y)
}

func orientation(p, q, r Point) int {
    val := (q.y-p.y)*(r.x-q.x) - (q.x-p.x)*(r.y-q.y)
    if val == 0 {
        return 0 // Collinear
    }
    if val > 0 {
        return 1 // Clockwise
    }
    return 2 // Counterclockwise
}

func distance(p, q Point) float64 {
    return math.Sqrt((p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y))
}

func grahamScan(points []Point) []Point {
    n := len(points)
    if n < 3 {
        return points
    }
    
    // Find bottom-most point (or leftmost in case of tie)
    ymin := points[0].y
    min := 0
    for i := 1; i < n; i++ {
        y := points[i].y
        if (y < ymin) || (ymin == y && points[i].x < points[min].x) {
            ymin = points[i].y
            min = i
        }
    }
    
    // Place bottom-most point at first position
    points[0], points[min] = points[min], points[0]
    
    // Sort points by polar angle with respect to p0
    p0 := points[0]
    sort.Slice(points[1:], func(i, j int) bool {
        i++
        j++
        o := orientation(p0, points[i], points[j])
        if o == 0 {
            return distance(p0, points[i]) < distance(p0, points[j])
        }
        return o == 2
    })
    
    // Create stack and push first three points
    stack := []Point{points[0], points[1], points[2]}
    
    // Process remaining points
    for i := 3; i < n; i++ {
        // Keep removing top while the angle formed by points next-to-top,
        // top, and points[i] makes a non-left turn
        for len(stack) > 1 && orientation(stack[len(stack)-2], stack[len(stack)-1], points[i]) != 2 {
            stack = stack[:len(stack)-1]
        }
        stack = append(stack, points[i])
    }
    
    return stack
}

func main() {
    points := []Point{
        {0, 3}, {1, 1}, {2, 2}, {4, 4},
        {0, 0}, {1, 2}, {3, 1}, {3, 3},
    }
    
    hull := grahamScan(points)
    
    fmt.Println("Convex Hull:")
    for _, point := range hull {
        fmt.Println(point)
    }
}
```

### 2. Line Intersection

#### Line Segment Intersection
```go
package main

import (
    "fmt"
    "math"
)

type Point struct {
    x, y float64
}

type Line struct {
    p1, p2 Point
}

func onSegment(p, q, r Point) bool {
    return q.x <= math.Max(p.x, r.x) && q.x >= math.Min(p.x, r.x) &&
           q.y <= math.Max(p.y, r.y) && q.y >= math.Min(p.y, r.y)
}

func orientation(p, q, r Point) int {
    val := (q.y-p.y)*(r.x-q.x) - (q.x-p.x)*(r.y-q.y)
    if val == 0 {
        return 0 // Collinear
    }
    if val > 0 {
        return 1 // Clockwise
    }
    return 2 // Counterclockwise
}

func doIntersect(l1, l2 Line) bool {
    o1 := orientation(l1.p1, l1.p2, l2.p1)
    o2 := orientation(l1.p1, l1.p2, l2.p2)
    o3 := orientation(l2.p1, l2.p2, l1.p1)
    o4 := orientation(l2.p1, l2.p2, l1.p2)
    
    // General case
    if o1 != o2 && o3 != o4 {
        return true
    }
    
    // Special cases
    if o1 == 0 && onSegment(l1.p1, l2.p1, l1.p2) {
        return true
    }
    
    if o2 == 0 && onSegment(l1.p1, l2.p2, l1.p2) {
        return true
    }
    
    if o3 == 0 && onSegment(l2.p1, l1.p1, l2.p2) {
        return true
    }
    
    if o4 == 0 && onSegment(l2.p1, l1.p2, l2.p2) {
        return true
    }
    
    return false
}

func main() {
    l1 := Line{Point{1, 1}, Point{10, 1}}
    l2 := Line{Point{1, 2}, Point{10, 2}}
    
    if doIntersect(l1, l2) {
        fmt.Println("Lines intersect")
    } else {
        fmt.Println("Lines do not intersect")
    }
}
```

## Number Theory

### 1. Prime Number Algorithms

#### Sieve of Eratosthenes
```go
package main

import "fmt"

func sieveOfEratosthenes(n int) []int {
    isPrime := make([]bool, n+1)
    for i := 2; i <= n; i++ {
        isPrime[i] = true
    }
    
    for p := 2; p*p <= n; p++ {
        if isPrime[p] {
            for i := p * p; i <= n; i += p {
                isPrime[i] = false
            }
        }
    }
    
    var primes []int
    for i := 2; i <= n; i++ {
        if isPrime[i] {
            primes = append(primes, i)
        }
    }
    
    return primes
}

func main() {
    n := 30
    primes := sieveOfEratosthenes(n)
    fmt.Printf("Prime numbers up to %d: %v\n", n, primes)
}
```

#### Extended Euclidean Algorithm
```go
package main

import "fmt"

func extendedGCD(a, b int) (int, int, int) {
    if a == 0 {
        return b, 0, 1
    }
    
    gcd, x1, y1 := extendedGCD(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return gcd, x, y
}

func modInverse(a, m int) int {
    gcd, x, _ := extendedGCD(a, m)
    if gcd != 1 {
        return -1 // Inverse doesn't exist
    }
    return (x%m + m) % m
}

func main() {
    a, b := 56, 15
    gcd, x, y := extendedGCD(a, b)
    fmt.Printf("GCD(%d, %d) = %d\n", a, b, gcd)
    fmt.Printf("x = %d, y = %d\n", x, y)
    fmt.Printf("%d * %d + %d * %d = %d\n", a, x, b, y, a*x+b*y)
    
    // Modular inverse
    a, m := 3, 11
    inv := modInverse(a, m)
    if inv != -1 {
        fmt.Printf("Modular inverse of %d mod %d is %d\n", a, m, inv)
    } else {
        fmt.Printf("Modular inverse of %d mod %d doesn't exist\n", a, m)
    }
}
```

## Combinatorics

### 1. Permutations and Combinations

#### Permutation Generation
```go
package main

import "fmt"

func permute(nums []int) [][]int {
    var result [][]int
    var backtrack func([]int, []int)
    
    backtrack = func(current, remaining []int) {
        if len(remaining) == 0 {
            result = append(result, append([]int(nil), current...))
            return
        }
        
        for i := 0; i < len(remaining); i++ {
            newCurrent := append(current, remaining[i])
            newRemaining := append(remaining[:i], remaining[i+1:]...)
            backtrack(newCurrent, newRemaining)
        }
    }
    
    backtrack([]int{}, nums)
    return result
}

func combinations(n, r int) [][]int {
    if r > n || r < 0 {
        return [][]int{}
    }
    
    if r == 0 {
        return [][]int{{}}
    }
    
    if r == n {
        return [][]int{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}[:n]}
    }
    
    // Generate combinations using bit manipulation
    var result [][]int
    for i := 0; i < (1 << n); i++ {
        if countBits(i) == r {
            var combo []int
            for j := 0; j < n; j++ {
                if (i>>j)&1 == 1 {
                    combo = append(combo, j)
                }
            }
            result = append(result, combo)
        }
    }
    
    return result
}

func countBits(n int) int {
    count := 0
    for n > 0 {
        count += n & 1
        n >>= 1
    }
    return count
}

func main() {
    // Permutations
    nums := []int{1, 2, 3}
    perms := permute(nums)
    fmt.Printf("Permutations of %v:\n", nums)
    for _, perm := range perms {
        fmt.Println(perm)
    }
    
    // Combinations
    n, r := 4, 2
    combos := combinations(n, r)
    fmt.Printf("\nCombinations of %d choose %d:\n", n, r)
    for _, combo := range combos {
        fmt.Println(combo)
    }
}
```

## Game Theory

### 1. Minimax Algorithm

#### Tic-Tac-Toe AI
```go
package main

import (
    "fmt"
    "math"
)

type Player int

const (
    EMPTY Player = iota
    X
    O
)

type Board [3][3]Player

func (b Board) isFull() bool {
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if b[i][j] == EMPTY {
                return false
            }
        }
    }
    return true
}

func (b Board) checkWinner() Player {
    // Check rows
    for i := 0; i < 3; i++ {
        if b[i][0] != EMPTY && b[i][0] == b[i][1] && b[i][1] == b[i][2] {
            return b[i][0]
        }
    }
    
    // Check columns
    for j := 0; j < 3; j++ {
        if b[0][j] != EMPTY && b[0][j] == b[1][j] && b[1][j] == b[2][j] {
            return b[0][j]
        }
    }
    
    // Check diagonals
    if b[0][0] != EMPTY && b[0][0] == b[1][1] && b[1][1] == b[2][2] {
        return b[0][0]
    }
    
    if b[0][2] != EMPTY && b[0][2] == b[1][1] && b[1][1] == b[2][0] {
        return b[0][2]
    }
    
    return EMPTY
}

func (b Board) isGameOver() bool {
    return b.checkWinner() != EMPTY || b.isFull()
}

func (b Board) evaluate() int {
    winner := b.checkWinner()
    switch winner {
    case X:
        return 10
    case O:
        return -10
    default:
        return 0
    }
}

func minimax(board Board, depth int, isMaximizing bool) int {
    if board.isGameOver() {
        return board.evaluate()
    }
    
    if isMaximizing {
        maxEval := math.MinInt32
        for i := 0; i < 3; i++ {
            for j := 0; j < 3; j++ {
                if board[i][j] == EMPTY {
                    board[i][j] = X
                    eval := minimax(board, depth+1, false)
                    board[i][j] = EMPTY
                    maxEval = max(maxEval, eval)
                }
            }
        }
        return maxEval
    } else {
        minEval := math.MaxInt32
        for i := 0; i < 3; i++ {
            for j := 0; j < 3; j++ {
                if board[i][j] == EMPTY {
                    board[i][j] = O
                    eval := minimax(board, depth+1, true)
                    board[i][j] = EMPTY
                    minEval = min(minEval, eval)
                }
            }
        }
        return minEval
    }
}

func findBestMove(board Board) (int, int) {
    bestVal := math.MinInt32
    bestMove := struct{ row, col int }{-1, -1}
    
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if board[i][j] == EMPTY {
                board[i][j] = X
                moveVal := minimax(board, 0, false)
                board[i][j] = EMPTY
                
                if moveVal > bestVal {
                    bestMove.row = i
                    bestMove.col = j
                    bestVal = moveVal
                }
            }
        }
    }
    
    return bestMove.row, bestMove.col
}

func (b Board) print() {
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            switch b[i][j] {
            case X:
                fmt.Print("X ")
            case O:
                fmt.Print("O ")
            default:
                fmt.Print("- ")
            }
        }
        fmt.Println()
    }
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    board := Board{}
    
    // Example game
    board[0][0] = O
    board[1][1] = X
    board[0][2] = O
    
    fmt.Println("Current board:")
    board.print()
    
    row, col := findBestMove(board)
    fmt.Printf("Best move for X: (%d, %d)\n", row, col)
}
```

## Follow-up Questions

### 1. Graph Algorithms
**Q: What's the time complexity of the Ford-Fulkerson algorithm?**
A: O(E * max_flow) where E is the number of edges and max_flow is the maximum flow value.

### 2. String Algorithms
**Q: How does the KMP algorithm improve over naive string matching?**
A: KMP avoids redundant comparisons by using information from previous matches, achieving O(n+m) time complexity.

### 3. Computational Geometry
**Q: What's the time complexity of Graham's scan for convex hull?**
A: O(n log n) due to the sorting step, where n is the number of points.

## Sources

### Books
- **Introduction to Algorithms** by Cormen, Leiserson, Rivest, and Stein
- **Algorithm Design** by Kleinberg and Tardos
- **Computational Geometry** by de Berg, Cheong, van Kreveld, and Overmars

### Online Resources
- **Competitive Programming** - Algorithm implementations
- **GeeksforGeeks** - Algorithm explanations
- **LeetCode** - Practice problems

## Projects

### 1. Graph Algorithm Library
**Objective**: Implement a comprehensive graph algorithm library
**Requirements**: Various graph algorithms, performance optimization
**Deliverables**: Complete library with documentation

### 2. String Processing Tool
**Objective**: Build a string processing tool with advanced algorithms
**Requirements**: Pattern matching, text analysis, performance
**Deliverables**: Production-ready string processing tool

### 3. Computational Geometry System
**Objective**: Create a computational geometry system
**Requirements**: Geometric algorithms, visualization, optimization
**Deliverables**: Complete geometry processing system

---

**Next**: [Performance Engineering](performance-engineering/README.md/) | **Previous**: [Cloud Architecture](cloud-architecture/README.md/) | **Up**: [Phase 2](README.md/)

