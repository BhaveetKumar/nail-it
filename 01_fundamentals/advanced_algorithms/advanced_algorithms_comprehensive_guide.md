# 🧮 Advanced Algorithms Comprehensive Guide

## Table of Contents
1. [Graph Algorithms](#graph-algorithms)
2. [Dynamic Programming](#dynamic-programming)
3. [String Algorithms](#string-algorithms)
4. [Computational Geometry](#computational-geometry)
5. [Number Theory Algorithms](#number-theory-algorithms)
6. [Advanced Data Structures](#advanced-data-structures)
7. [Optimization Algorithms](#optimization-algorithms)
8. [Parallel Algorithms](#parallel-algorithms)
9. [Go Implementation Examples](#go-implementation-examples)
10. [Interview Questions](#interview-questions)

## Graph Algorithms

### Advanced Graph Traversal

```go
package main

import (
    "fmt"
    "math"
)

type Graph struct {
    Vertices int
    Edges    [][]Edge
}

type Edge struct {
    To     int
    Weight int
}

func NewGraph(vertices int) *Graph {
    edges := make([][]Edge, vertices)
    for i := range edges {
        edges[i] = make([]Edge, 0)
    }
    return &Graph{Vertices: vertices, Edges: edges}
}

func (g *Graph) AddEdge(from, to, weight int) {
    g.Edges[from] = append(g.Edges[from], Edge{To: to, Weight: weight})
}

// Dijkstra's Algorithm with Priority Queue
func (g *Graph) Dijkstra(start int) []int {
    dist := make([]int, g.Vertices)
    for i := range dist {
        dist[i] = math.MaxInt32
    }
    dist[start] = 0
    
    visited := make([]bool, g.Vertices)
    
    for i := 0; i < g.Vertices; i++ {
        u := g.minDistance(dist, visited)
        visited[u] = true
        
        for _, edge := range g.Edges[u] {
            if !visited[edge.To] && dist[u] != math.MaxInt32 {
                if dist[u]+edge.Weight < dist[edge.To] {
                    dist[edge.To] = dist[u] + edge.Weight
                }
            }
        }
    }
    
    return dist
}

func (g *Graph) minDistance(dist []int, visited []bool) int {
    min := math.MaxInt32
    minIndex := -1
    
    for v := 0; v < g.Vertices; v++ {
        if !visited[v] && dist[v] <= min {
            min = dist[v]
            minIndex = v
        }
    }
    
    return minIndex
}

// Bellman-Ford Algorithm
func (g *Graph) BellmanFord(start int) ([]int, bool) {
    dist := make([]int, g.Vertices)
    for i := range dist {
        dist[i] = math.MaxInt32
    }
    dist[start] = 0
    
    // Relax edges V-1 times
    for i := 0; i < g.Vertices-1; i++ {
        for u := 0; u < g.Vertices; u++ {
            for _, edge := range g.Edges[u] {
                if dist[u] != math.MaxInt32 && dist[u]+edge.Weight < dist[edge.To] {
                    dist[edge.To] = dist[u] + edge.Weight
                }
            }
        }
    }
    
    // Check for negative cycles
    for u := 0; u < g.Vertices; u++ {
        for _, edge := range g.Edges[u] {
            if dist[u] != math.MaxInt32 && dist[u]+edge.Weight < dist[edge.To] {
                return dist, false // Negative cycle detected
            }
        }
    }
    
    return dist, true
}

// Floyd-Warshall Algorithm
func (g *Graph) FloydWarshall() [][]int {
    dist := make([][]int, g.Vertices)
    for i := range dist {
        dist[i] = make([]int, g.Vertices)
        for j := range dist[i] {
            if i == j {
                dist[i][j] = 0
            } else {
                dist[i][j] = math.MaxInt32
            }
        }
    }
    
    // Initialize with edge weights
    for u := 0; u < g.Vertices; u++ {
        for _, edge := range g.Edges[u] {
            dist[u][edge.To] = edge.Weight
        }
    }
    
    // Floyd-Warshall algorithm
    for k := 0; k < g.Vertices; k++ {
        for i := 0; i < g.Vertices; i++ {
            for j := 0; j < g.Vertices; j++ {
                if dist[i][k] != math.MaxInt32 && dist[k][j] != math.MaxInt32 {
                    if dist[i][k]+dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j]
                    }
                }
            }
        }
    }
    
    return dist
}
```

### Minimum Spanning Tree

```go
// Kruskal's Algorithm
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

type Edge struct {
    From   int
    To     int
    Weight int
}

func (g *Graph) KruskalMST() []Edge {
    var edges []Edge
    for u := 0; u < g.Vertices; u++ {
        for _, edge := range g.Edges[u] {
            edges = append(edges, Edge{From: u, To: edge.To, Weight: edge.Weight})
        }
    }
    
    // Sort edges by weight
    sort.Slice(edges, func(i, j int) bool {
        return edges[i].Weight < edges[j].Weight
    })
    
    uf := NewUnionFind(g.Vertices)
    var mst []Edge
    
    for _, edge := range edges {
        if uf.Union(edge.From, edge.To) {
            mst = append(mst, edge)
            if len(mst) == g.Vertices-1 {
                break
            }
        }
    }
    
    return mst
}

// Prim's Algorithm
func (g *Graph) PrimMST() []Edge {
    parent := make([]int, g.Vertices)
    key := make([]int, g.Vertices)
    mstSet := make([]bool, g.Vertices)
    
    for i := range key {
        key[i] = math.MaxInt32
    }
    
    key[0] = 0
    parent[0] = -1
    
    for count := 0; count < g.Vertices-1; count++ {
        u := g.minKey(key, mstSet)
        mstSet[u] = true
        
        for _, edge := range g.Edges[u] {
            if !mstSet[edge.To] && edge.Weight < key[edge.To] {
                parent[edge.To] = u
                key[edge.To] = edge.Weight
            }
        }
    }
    
    var mst []Edge
    for i := 1; i < g.Vertices; i++ {
        mst = append(mst, Edge{From: parent[i], To: i, Weight: key[i]})
    }
    
    return mst
}

func (g *Graph) minKey(key []int, mstSet []bool) int {
    min := math.MaxInt32
    minIndex := -1
    
    for v := 0; v < g.Vertices; v++ {
        if !mstSet[v] && key[v] < min {
            min = key[v]
            minIndex = v
        }
    }
    
    return minIndex
}
```

## Dynamic Programming

### Advanced DP Patterns

```go
// Longest Common Subsequence
func LCS(text1, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    return dp[m][n]
}

// Edit Distance (Levenshtein Distance)
func EditDistance(word1, word2 string) int {
    m, n := len(word1), len(word2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    for i := 0; i <= m; i++ {
        dp[i][0] = i
    }
    for j := 0; j <= n; j++ {
        dp[0][j] = j
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if word1[i-1] == word2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
    }
    
    return dp[m][n]
}

// Longest Increasing Subsequence
func LIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    dp := make([]int, len(nums))
    for i := range dp {
        dp[i] = 1
    }
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
    }
    
    maxLen := 0
    for _, length := range dp {
        maxLen = max(maxLen, length)
    }
    
    return maxLen
}

// 0/1 Knapsack Problem
func Knapsack(weights, values []int, capacity int) int {
    n := len(weights)
    dp := make([][]int, n+1)
    for i := range dp {
        dp[i] = make([]int, capacity+1)
    }
    
    for i := 1; i <= n; i++ {
        for w := 1; w <= capacity; w++ {
            if weights[i-1] <= w {
                dp[i][w] = max(values[i-1]+dp[i-1][w-weights[i-1]], dp[i-1][w])
            } else {
                dp[i][w] = dp[i-1][w]
            }
        }
    }
    
    return dp[n][capacity]
}

// Coin Change Problem
func CoinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for i := range dp {
        dp[i] = amount + 1
    }
    dp[0] = 0
    
    for i := 1; i <= amount; i++ {
        for _, coin := range coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i-coin]+1)
            }
        }
    }
    
    if dp[amount] > amount {
        return -1
    }
    return dp[amount]
}
```

## String Algorithms

### Advanced String Matching

```go
// KMP Algorithm
func KMP(text, pattern string) []int {
    var matches []int
    n, m := len(text), len(pattern)
    
    if m == 0 {
        return matches
    }
    
    // Build failure function
    fail := make([]int, m)
    j := 0
    for i := 1; i < m; i++ {
        for j > 0 && pattern[i] != pattern[j] {
            j = fail[j-1]
        }
        if pattern[i] == pattern[j] {
            j++
        }
        fail[i] = j
    }
    
    // Search
    j = 0
    for i := 0; i < n; i++ {
        for j > 0 && text[i] != pattern[j] {
            j = fail[j-1]
        }
        if text[i] == pattern[j] {
            j++
        }
        if j == m {
            matches = append(matches, i-m+1)
            j = fail[j-1]
        }
    }
    
    return matches
}

// Rabin-Karp Algorithm
func RabinKarp(text, pattern string) []int {
    var matches []int
    n, m := len(text), len(pattern)
    
    if m == 0 || m > n {
        return matches
    }
    
    const base = 256
    const mod = 1000000007
    
    // Calculate hash of pattern
    patternHash := 0
    for i := 0; i < m; i++ {
        patternHash = (patternHash*base + int(pattern[i])) % mod
    }
    
    // Calculate hash of first window
    textHash := 0
    for i := 0; i < m; i++ {
        textHash = (textHash*base + int(text[i])) % mod
    }
    
    // Calculate base^(m-1) for rolling hash
    h := 1
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    // Slide the pattern over text
    for i := 0; i <= n-m; i++ {
        if patternHash == textHash {
            // Check character by character
            match := true
            for j := 0; j < m; j++ {
                if text[i+j] != pattern[j] {
                    match = false
                    break
                }
            }
            if match {
                matches = append(matches, i)
            }
        }
        
        // Calculate hash for next window
        if i < n-m {
            textHash = (base*(textHash-int(text[i])*h) + int(text[i+m])) % mod
            if textHash < 0 {
                textHash += mod
            }
        }
    }
    
    return matches
}

// Z Algorithm
func ZAlgorithm(text string) []int {
    n := len(text)
    z := make([]int, n)
    z[0] = n
    
    l, r := 0, 0
    for i := 1; i < n; i++ {
        if i <= r {
            z[i] = min(r-i+1, z[i-l])
        }
        
        for i+z[i] < n && text[z[i]] == text[i+z[i]] {
            z[i]++
        }
        
        if i+z[i]-1 > r {
            l, r = i, i+z[i]-1
        }
    }
    
    return z
}

// Suffix Array
func SuffixArray(text string) []int {
    n := len(text)
    suffixes := make([]int, n)
    for i := range suffixes {
        suffixes[i] = i
    }
    
    // Sort suffixes by their first character
    sort.Slice(suffixes, func(i, j int) bool {
        return text[suffixes[i]] < text[suffixes[j]]
    })
    
    // Sort by longer prefixes
    for k := 1; k < n; k *= 2 {
        // Create equivalence classes
        classes := make([]int, n)
        class := 0
        for i := 1; i < n; i++ {
            if text[suffixes[i]] != text[suffixes[i-1]] {
                class++
            }
            classes[suffixes[i]] = class
        }
        
        // Sort by first k characters
        sort.Slice(suffixes, func(i, j int) bool {
            a, b := suffixes[i], suffixes[j]
            if classes[a] != classes[b] {
                return classes[a] < classes[b]
            }
            a += k
            b += k
            if a >= n || b >= n {
                return a > b
            }
            return classes[a] < classes[b]
        })
        
        // Update classes
        newClasses := make([]int, n)
        class = 0
        for i := 1; i < n; i++ {
            a, b := suffixes[i], suffixes[i-1]
            if classes[a] != classes[b] || 
               (a+k < n && b+k < n && classes[a+k] != classes[b+k]) {
                class++
            }
            newClasses[suffixes[i]] = class
        }
        classes = newClasses
    }
    
    return suffixes
}
```

## Computational Geometry

### Geometric Algorithms

```go
type Point struct {
    X, Y float64
}

type Line struct {
    A, B, C float64 // Ax + By + C = 0
}

// Convex Hull using Graham Scan
func ConvexHull(points []Point) []Point {
    if len(points) < 3 {
        return points
    }
    
    // Find bottom-most point
    bottom := 0
    for i := 1; i < len(points); i++ {
        if points[i].Y < points[bottom].Y || 
           (points[i].Y == points[bottom].Y && points[i].X < points[bottom].X) {
            bottom = i
        }
    }
    
    // Swap bottom point to beginning
    points[0], points[bottom] = points[bottom], points[0]
    
    // Sort by polar angle
    sort.Slice(points[1:], func(i, j int) bool {
        p1, p2 := points[i+1], points[j+1]
        cross := crossProduct(points[0], p1, p2)
        if cross == 0 {
            return distance(points[0], p1) < distance(points[0], p2)
        }
        return cross > 0
    })
    
    // Build convex hull
    hull := []Point{points[0], points[1]}
    
    for i := 2; i < len(points); i++ {
        for len(hull) > 1 && 
              crossProduct(hull[len(hull)-2], hull[len(hull)-1], points[i]) <= 0 {
            hull = hull[:len(hull)-1]
        }
        hull = append(hull, points[i])
    }
    
    return hull
}

func crossProduct(o, a, b Point) float64 {
    return (a.X-o.X)*(b.Y-o.Y) - (a.Y-o.Y)*(b.X-o.X)
}

func distance(a, b Point) float64 {
    dx := a.X - b.X
    dy := a.Y - b.Y
    return dx*dx + dy*dy
}

// Line Intersection
func LineIntersection(l1, l2 Line) (Point, bool) {
    det := l1.A*l2.B - l2.A*l1.B
    if math.Abs(det) < 1e-10 {
        return Point{}, false // Lines are parallel
    }
    
    x := (l1.B*l2.C - l2.B*l1.C) / det
    y := (l2.A*l1.C - l1.A*l2.C) / det
    
    return Point{X: x, Y: y}, true
}

// Point in Polygon
func PointInPolygon(point Point, polygon []Point) bool {
    n := len(polygon)
    if n < 3 {
        return false
    }
    
    inside := false
    j := n - 1
    
    for i := 0; i < n; i++ {
        if ((polygon[i].Y > point.Y) != (polygon[j].Y > point.Y)) &&
           (point.X < (polygon[j].X-polygon[i].X)*(point.Y-polygon[i].Y)/(polygon[j].Y-polygon[i].Y)+polygon[i].X) {
            inside = !inside
        }
        j = i
    }
    
    return inside
}

// Closest Pair of Points
func ClosestPair(points []Point) (Point, Point, float64) {
    if len(points) < 2 {
        return Point{}, Point{}, math.Inf(1)
    }
    
    // Sort by x-coordinate
    sorted := make([]Point, len(points))
    copy(sorted, points)
    sort.Slice(sorted, func(i, j int) bool {
        return sorted[i].X < sorted[j].X
    })
    
    return closestPairRec(sorted)
}

func closestPairRec(points []Point) (Point, Point, float64) {
    n := len(points)
    if n <= 3 {
        return bruteForceClosestPair(points)
    }
    
    mid := n / 2
    midPoint := points[mid]
    
    left := points[:mid]
    right := points[mid:]
    
    p1, q1, d1 := closestPairRec(left)
    p2, q2, d2 := closestPairRec(right)
    
    var minDist float64
    var p, q Point
    
    if d1 < d2 {
        minDist = d1
        p, q = p1, q1
    } else {
        minDist = d2
        p, q = p2, q2
    }
    
    // Check points near the dividing line
    strip := make([]Point, 0)
    for _, point := range points {
        if math.Abs(point.X-midPoint.X) < minDist {
            strip = append(strip, point)
        }
    }
    
    p3, q3, d3 := closestInStrip(strip, minDist)
    if d3 < minDist {
        return p3, q3, d3
    }
    
    return p, q, minDist
}

func bruteForceClosestPair(points []Point) (Point, Point, float64) {
    minDist := math.Inf(1)
    var p1, p2 Point
    
    for i := 0; i < len(points); i++ {
        for j := i + 1; j < len(points); j++ {
            dist := distance(points[i], points[j])
            if dist < minDist {
                minDist = dist
                p1, p2 = points[i], points[j]
            }
        }
    }
    
    return p1, p2, minDist
}

func closestInStrip(strip []Point, minDist float64) (Point, Point, float64) {
    // Sort by y-coordinate
    sort.Slice(strip, func(i, j int) bool {
        return strip[i].Y < strip[j].Y
    })
    
    minDistStrip := minDist
    var p1, p2 Point
    
    for i := 0; i < len(strip); i++ {
        for j := i + 1; j < len(strip) && strip[j].Y-strip[i].Y < minDistStrip; j++ {
            dist := distance(strip[i], strip[j])
            if dist < minDistStrip {
                minDistStrip = dist
                p1, p2 = strip[i], strip[j]
            }
        }
    }
    
    return p1, p2, minDistStrip
}
```

## Number Theory Algorithms

### Advanced Number Theory

```go
// Extended Euclidean Algorithm
func ExtendedGCD(a, b int) (int, int, int) {
    if a == 0 {
        return b, 0, 1
    }
    
    gcd, x1, y1 := ExtendedGCD(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return gcd, x, y
}

// Modular Exponentiation
func ModExp(base, exponent, modulus int) int {
    result := 1
    base = base % modulus
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % modulus
        }
        exponent = exponent >> 1
        base = (base * base) % modulus
    }
    
    return result
}

// Miller-Rabin Primality Test
func IsPrime(n int) bool {
    if n < 2 {
        return false
    }
    if n == 2 || n == 3 {
        return true
    }
    if n%2 == 0 {
        return false
    }
    
    // Write n-1 as d * 2^r
    d := n - 1
    r := 0
    for d%2 == 0 {
        d /= 2
        r++
    }
    
    // Test with bases
    bases := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    
    for _, a := range bases {
        if a >= n {
            continue
        }
        
        x := ModExp(a, d, n)
        if x == 1 || x == n-1 {
            continue
        }
        
        composite := true
        for i := 0; i < r-1; i++ {
            x = (x * x) % n
            if x == n-1 {
                composite = false
                break
            }
        }
        
        if composite {
            return false
        }
    }
    
    return true
}

// Pollard's Rho Algorithm for Factorization
func PollardRho(n int) int {
    if n%2 == 0 {
        return 2
    }
    
    x := 2
    y := 2
    d := 1
    
    f := func(x int) int {
        return (x*x + 1) % n
    }
    
    for d == 1 {
        x = f(x)
        y = f(f(y))
        d = gcd(abs(x-y), n)
    }
    
    if d == n {
        return PollardRho(n)
    }
    
    return d
}

func gcd(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}

// Chinese Remainder Theorem
func ChineseRemainderTheorem(remainders, moduli []int) (int, error) {
    if len(remainders) != len(moduli) {
        return 0, fmt.Errorf("lengths don't match")
    }
    
    n := len(remainders)
    if n == 0 {
        return 0, nil
    }
    
    // Calculate product of all moduli
    product := 1
    for _, m := range moduli {
        product *= m
    }
    
    result := 0
    for i := 0; i < n; i++ {
        // Calculate Mi = product / moduli[i]
        Mi := product / moduli[i]
        
        // Calculate Mi^(-1) mod moduli[i]
        _, MiInv, _ := ExtendedGCD(Mi, moduli[i])
        if MiInv < 0 {
            MiInv += moduli[i]
        }
        
        result += remainders[i] * Mi * MiInv
    }
    
    return result % product, nil
}
```

## Advanced Data Structures

### Segment Tree

```go
type SegmentTree struct {
    tree []int
    n    int
}

func NewSegmentTree(arr []int) *SegmentTree {
    n := len(arr)
    tree := make([]int, 4*n)
    
    st := &SegmentTree{tree: tree, n: n}
    st.build(arr, 1, 0, n-1)
    return st
}

func (st *SegmentTree) build(arr []int, node, start, end int) {
    if start == end {
        st.tree[node] = arr[start]
    } else {
        mid := (start + end) / 2
        st.build(arr, 2*node, start, mid)
        st.build(arr, 2*node+1, mid+1, end)
        st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
    }
}

func (st *SegmentTree) Query(l, r int) int {
    return st.query(1, 0, st.n-1, l, r)
}

func (st *SegmentTree) query(node, start, end, l, r int) int {
    if r < start || end < l {
        return 0
    }
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    mid := (start + end) / 2
    left := st.query(2*node, start, mid, l, r)
    right := st.query(2*node+1, mid+1, end, l, r)
    return left + right
}

func (st *SegmentTree) Update(idx, val int) {
    st.update(1, 0, st.n-1, idx, val)
}

func (st *SegmentTree) update(node, start, end, idx, val int) {
    if start == end {
        st.tree[node] = val
    } else {
        mid := (start + end) / 2
        if idx <= mid {
            st.update(2*node, start, mid, idx, val)
        } else {
            st.update(2*node+1, mid+1, end, idx, val)
        }
        st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
    }
}
```

### Fenwick Tree (Binary Indexed Tree)

```go
type FenwickTree struct {
    tree []int
    n    int
}

func NewFenwickTree(size int) *FenwickTree {
    return &FenwickTree{
        tree: make([]int, size+1),
        n:    size,
    }
}

func (ft *FenwickTree) Update(idx, val int) {
    for idx <= ft.n {
        ft.tree[idx] += val
        idx += idx & (-idx)
    }
}

func (ft *FenwickTree) Query(idx int) int {
    sum := 0
    for idx > 0 {
        sum += ft.tree[idx]
        idx -= idx & (-idx)
    }
    return sum
}

func (ft *FenwickTree) RangeQuery(l, r int) int {
    return ft.Query(r) - ft.Query(l-1)
}
```

## Optimization Algorithms

### Simulated Annealing

```go
func SimulatedAnnealing(initialState []int, costFunc func([]int) float64, 
                       neighborFunc func([]int) []int, 
                       initialTemp, finalTemp, coolingRate float64) []int {
    current := make([]int, len(initialState))
    copy(current, initialState)
    best := make([]int, len(initialState))
    copy(best, current)
    
    currentCost := costFunc(current)
    bestCost := currentCost
    
    temp := initialTemp
    
    for temp > finalTemp {
        neighbor := neighborFunc(current)
        neighborCost := costFunc(neighbor)
        
        delta := neighborCost - currentCost
        
        if delta < 0 || math.Exp(-delta/temp) > rand.Float64() {
            current = neighbor
            currentCost = neighborCost
            
            if currentCost < bestCost {
                copy(best, current)
                bestCost = currentCost
            }
        }
        
        temp *= coolingRate
    }
    
    return best
}
```

## Parallel Algorithms

### Parallel Merge Sort

```go
func ParallelMergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    
    if len(arr) < 1000 {
        return SequentialMergeSort(arr)
    }
    
    mid := len(arr) / 2
    
    var left, right []int
    var wg sync.WaitGroup
    
    wg.Add(2)
    
    go func() {
        defer wg.Done()
        left = ParallelMergeSort(arr[:mid])
    }()
    
    go func() {
        defer wg.Done()
        right = ParallelMergeSort(arr[mid:])
    }()
    
    wg.Wait()
    
    return merge(left, right)
}

func SequentialMergeSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    
    mid := len(arr) / 2
    left := SequentialMergeSort(arr[:mid])
    right := SequentialMergeSort(arr[mid:])
    
    return merge(left, right)
}

func merge(left, right []int) []int {
    result := make([]int, len(left)+len(right))
    i, j, k := 0, 0, 0
    
    for i < len(left) && j < len(right) {
        if left[i] <= right[j] {
            result[k] = left[i]
            i++
        } else {
            result[k] = right[j]
            j++
        }
        k++
    }
    
    for i < len(left) {
        result[k] = left[i]
        i++
        k++
    }
    
    for j < len(right) {
        result[k] = right[j]
        j++
        k++
    }
    
    return result
}
```

## Interview Questions

### Basic Concepts
1. **What is the time complexity of Dijkstra's algorithm?**
2. **Explain the difference between BFS and DFS.**
3. **What is dynamic programming?**
4. **How does the KMP algorithm work?**
5. **What is the purpose of a segment tree?**

### Advanced Topics
1. **How would you implement a suffix array?**
2. **Explain the Miller-Rabin primality test.**
3. **How does the Floyd-Warshall algorithm work?**
4. **What is the Chinese Remainder Theorem?**
5. **How would you implement a parallel sorting algorithm?**

### System Design
1. **Design an algorithm to find the shortest path in a weighted graph.**
2. **How would you implement a text search engine?**
3. **Design a system to find the closest pair of points.**
4. **How would you implement a distributed sorting algorithm?**
5. **Design a system to factorize large numbers.**

## Conclusion

Advanced algorithms are essential for solving complex problems efficiently. Key areas to master:

- **Graph Algorithms**: Shortest paths, minimum spanning trees, network flow
- **Dynamic Programming**: Optimization problems, sequence alignment
- **String Algorithms**: Pattern matching, text processing
- **Computational Geometry**: Convex hulls, line intersections
- **Number Theory**: Prime testing, factorization, modular arithmetic
- **Advanced Data Structures**: Segment trees, Fenwick trees
- **Optimization**: Simulated annealing, genetic algorithms
- **Parallel Algorithms**: Concurrent processing, distributed computing

Understanding these algorithms helps in:
- Solving complex algorithmic problems
- Optimizing system performance
- Designing efficient data structures
- Preparing for technical interviews
- Building scalable systems

Practice implementing these algorithms and understand their time and space complexities to become a proficient algorithm engineer.
