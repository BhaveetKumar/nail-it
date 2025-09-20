# 🧮 Advanced Algorithms Comprehensive Guide

> **Master advanced algorithms and data structures for technical interviews and real-world applications**

## 📚 Table of Contents

1. [Advanced Data Structures](#-advanced-data-structures)
2. [Graph Algorithms](#-graph-algorithms)
3. [Dynamic Programming](#-dynamic-programming)
4. [String Algorithms](#-string-algorithms)
5. [Mathematical Algorithms](#-mathematical-algorithms)
6. [Computational Geometry](#-computational-geometry)
7. [Advanced Sorting & Searching](#-advanced-sorting--searching)
8. [Concurrent Algorithms](#-concurrent-algorithms)
9. [Algorithm Design Patterns](#-algorithm-design-patterns)
10. [Performance Analysis](#-performance-analysis)

---

## 🏗️ Advanced Data Structures

### 1. Segment Trees

**Purpose**: Range queries and updates in O(log n) time

```go
package main

import (
    "fmt"
    "math"
)

type SegmentTree struct {
    tree   []int
    size   int
    data   []int
}

func NewSegmentTree(data []int) *SegmentTree {
    n := len(data)
    size := 1
    for size < n {
        size <<= 1
    }
    
    tree := make([]int, 2*size)
    st := &SegmentTree{
        tree: tree,
        size: size,
        data: data,
    }
    
    st.build(1, 0, n-1)
    return st
}

func (st *SegmentTree) build(node, start, end int) {
    if start == end {
        st.tree[node] = st.data[start]
    } else {
        mid := (start + end) / 2
        st.build(2*node, start, mid)
        st.build(2*node+1, mid+1, end)
        st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
    }
}

func (st *SegmentTree) Query(l, r int) int {
    return st.query(1, 0, st.size-1, l, r)
}

func (st *SegmentTree) query(node, start, end, l, r int) int {
    if r < start || end < l {
        return 0
    }
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    mid := (start + end) / 2
    leftSum := st.query(2*node, start, mid, l, r)
    rightSum := st.query(2*node+1, mid+1, end, l, r)
    return leftSum + rightSum
}

func (st *SegmentTree) Update(pos, value int) {
    st.update(1, 0, st.size-1, pos, value)
}

func (st *SegmentTree) update(node, start, end, pos, value int) {
    if start == end {
        st.tree[node] = value
    } else {
        mid := (start + end) / 2
        if pos <= mid {
            st.update(2*node, start, mid, pos, value)
        } else {
            st.update(2*node+1, mid+1, end, pos, value)
        }
        st.tree[node] = st.tree[2*node] + st.tree[2*node+1]
    }
}

// Example usage
func main() {
    data := []int{1, 3, 5, 7, 9, 11}
    st := NewSegmentTree(data)
    
    fmt.Println("Sum from index 1 to 3:", st.Query(1, 3)) // 3 + 5 + 7 = 15
    st.Update(1, 10)
    fmt.Println("Sum from index 1 to 3 after update:", st.Query(1, 3)) // 10 + 5 + 7 = 22
}
```

### 2. Fenwick Tree (Binary Indexed Tree)

**Purpose**: Efficient prefix sum queries and updates

```go
type FenwickTree struct {
    tree []int
    size int
}

func NewFenwickTree(size int) *FenwickTree {
    return &FenwickTree{
        tree: make([]int, size+1),
        size: size,
    }
}

func (ft *FenwickTree) Update(index, delta int) {
    for i := index + 1; i <= ft.size; i += i & (-i) {
        ft.tree[i] += delta
    }
}

func (ft *FenwickTree) Query(index int) int {
    sum := 0
    for i := index + 1; i > 0; i -= i & (-i) {
        sum += ft.tree[i]
    }
    return sum
}

func (ft *FenwickTree) RangeQuery(l, r int) int {
    return ft.Query(r) - ft.Query(l-1)
}
```

### 3. Trie (Prefix Tree)

**Purpose**: Efficient string operations and prefix matching

```go
type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
    count    int
}

type Trie struct {
    root *TrieNode
}

func NewTrie() *Trie {
    return &Trie{
        root: &TrieNode{
            children: make(map[rune]*TrieNode),
        },
    }
}

func (t *Trie) Insert(word string) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{
                children: make(map[rune]*TrieNode),
            }
        }
        node = node.children[char]
        node.count++
    }
    node.isEnd = true
}

func (t *Trie) Search(word string) bool {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return false
        }
        node = node.children[char]
    }
    return true
}

func (t *Trie) CountPrefix(prefix string) int {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return 0
        }
        node = node.children[char]
    }
    return node.count
}
```

### 4. Disjoint Set Union (Union-Find)

**Purpose**: Efficiently manage disjoint sets with union and find operations

```go
type UnionFind struct {
    parent []int
    rank   []int
    count  int
}

func NewUnionFind(n int) *UnionFind {
    parent := make([]int, n)
    rank := make([]int, n)
    for i := 0; i < n; i++ {
        parent[i] = i
        rank[i] = 1
    }
    return &UnionFind{
        parent: parent,
        rank:   rank,
        count:  n,
    }
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        uf.parent[x] = uf.Find(uf.parent[x]) // Path compression
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) bool {
    rootX := uf.Find(x)
    rootY := uf.Find(y)
    
    if rootX == rootY {
        return false
    }
    
    // Union by rank
    if uf.rank[rootX] < uf.rank[rootY] {
        uf.parent[rootX] = rootY
    } else if uf.rank[rootX] > uf.rank[rootY] {
        uf.parent[rootY] = rootX
    } else {
        uf.parent[rootY] = rootX
        uf.rank[rootX]++
    }
    
    uf.count--
    return true
}

func (uf *UnionFind) Connected(x, y int) bool {
    return uf.Find(x) == uf.Find(y)
}

func (uf *UnionFind) Count() int {
    return uf.count
}
```

---

## 🌐 Graph Algorithms

### 1. Shortest Path Algorithms

#### Dijkstra's Algorithm

```go
type Edge struct {
    to     int
    weight int
}

type Graph struct {
    adj [][]Edge
}

func (g *Graph) Dijkstra(start int) []int {
    n := len(g.adj)
    dist := make([]int, n)
    for i := range dist {
        dist[i] = math.MaxInt32
    }
    dist[start] = 0
    
    pq := NewMinHeap()
    pq.Push(HeapItem{node: start, dist: 0})
    
    for !pq.IsEmpty() {
        item := pq.Pop()
        u := item.node
        d := item.dist
        
        if d > dist[u] {
            continue
        }
        
        for _, edge := range g.adj[u] {
            v := edge.to
            w := edge.weight
            
            if dist[u]+w < dist[v] {
                dist[v] = dist[u] + w
                pq.Push(HeapItem{node: v, dist: dist[v]})
            }
        }
    }
    
    return dist
}
```

#### Bellman-Ford Algorithm

```go
func (g *Graph) BellmanFord(start int) ([]int, bool) {
    n := len(g.adj)
    dist := make([]int, n)
    for i := range dist {
        dist[i] = math.MaxInt32
    }
    dist[start] = 0
    
    // Relax edges V-1 times
    for i := 0; i < n-1; i++ {
        for u := 0; u < n; u++ {
            for _, edge := range g.adj[u] {
                v := edge.to
                w := edge.weight
                if dist[u] != math.MaxInt32 && dist[u]+w < dist[v] {
                    dist[v] = dist[u] + w
                }
            }
        }
    }
    
    // Check for negative cycles
    for u := 0; u < n; u++ {
        for _, edge := range g.adj[u] {
            v := edge.to
            w := edge.weight
            if dist[u] != math.MaxInt32 && dist[u]+w < dist[v] {
                return nil, false // Negative cycle detected
            }
        }
    }
    
    return dist, true
}
```

### 2. Minimum Spanning Tree

#### Kruskal's Algorithm

```go
type Edge struct {
    from   int
    to     int
    weight int
}

func KruskalMST(edges []Edge, n int) []Edge {
    // Sort edges by weight
    sort.Slice(edges, func(i, j int) bool {
        return edges[i].weight < edges[j].weight
    })
    
    uf := NewUnionFind(n)
    mst := []Edge{}
    
    for _, edge := range edges {
        if uf.Union(edge.from, edge.to) {
            mst = append(mst, edge)
            if len(mst) == n-1 {
                break
            }
        }
    }
    
    return mst
}
```

#### Prim's Algorithm

```go
func PrimMST(graph [][]Edge, start int) []Edge {
    n := len(graph)
    visited := make([]bool, n)
    mst := []Edge{}
    
    pq := NewMinHeap()
    pq.Push(HeapItem{node: start, dist: 0})
    
    for !pq.IsEmpty() && len(mst) < n-1 {
        item := pq.Pop()
        u := item.node
        
        if visited[u] {
            continue
        }
        visited[u] = true
        
        for _, edge := range graph[u] {
            v := edge.to
            if !visited[v] {
                pq.Push(HeapItem{node: v, dist: edge.weight})
            }
        }
    }
    
    return mst
}
```

### 3. Topological Sorting

```go
func TopologicalSort(graph [][]int) []int {
    n := len(graph)
    inDegree := make([]int, n)
    
    // Calculate in-degrees
    for u := 0; u < n; u++ {
        for _, v := range graph[u] {
            inDegree[v]++
        }
    }
    
    // Find nodes with no incoming edges
    queue := []int{}
    for i := 0; i < n; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    result := []int{}
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        result = append(result, u)
        
        for _, v := range graph[u] {
            inDegree[v]--
            if inDegree[v] == 0 {
                queue = append(queue, v)
            }
        }
    }
    
    if len(result) != n {
        return nil // Cycle detected
    }
    
    return result
}
```

---

## 🔄 Dynamic Programming

### 1. Classic DP Problems

#### Longest Common Subsequence

```go
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

func LCSWithPath(text1, text2 string) string {
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
    
    // Reconstruct the LCS
    result := make([]byte, 0, dp[m][n])
    i, j := m, n
    for i > 0 && j > 0 {
        if text1[i-1] == text2[j-1] {
            result = append(result, text1[i-1])
            i--
            j--
        } else if dp[i-1][j] > dp[i][j-1] {
            i--
        } else {
            j--
        }
    }
    
    // Reverse the result
    for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
        result[i], result[j] = result[j], result[i]
    }
    
    return string(result)
}
```

#### Edit Distance (Levenshtein Distance)

```go
func EditDistance(word1, word2 string) int {
    m, n := len(word1), len(word2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    // Initialize base cases
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
                dp[i][j] = 1 + min(
                    dp[i-1][j],   // Delete
                    dp[i][j-1],   // Insert
                    dp[i-1][j-1], // Replace
                )
            }
        }
    }
    
    return dp[m][n]
}
```

#### Knapsack Problem

```go
func Knapsack(weights, values []int, capacity int) int {
    n := len(weights)
    dp := make([][]int, n+1)
    for i := range dp {
        dp[i] = make([]int, capacity+1)
    }
    
    for i := 1; i <= n; i++ {
        for w := 1; w <= capacity; w++ {
            if weights[i-1] <= w {
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]]+values[i-1],
                )
            } else {
                dp[i][w] = dp[i-1][w]
            }
        }
    }
    
    return dp[n][capacity]
}

// Space-optimized version
func KnapsackOptimized(weights, values []int, capacity int) int {
    dp := make([]int, capacity+1)
    
    for i := 0; i < len(weights); i++ {
        for w := capacity; w >= weights[i]; w-- {
            dp[w] = max(dp[w], dp[w-weights[i]]+values[i])
        }
    }
    
    return dp[capacity]
}
```

### 2. Advanced DP Patterns

#### State Machine DP

```go
func MaxProfit(prices []int) int {
    n := len(prices)
    if n < 2 {
        return 0
    }
    
    // States: 0 = no stock, 1 = have stock
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, 2)
    }
    
    dp[0][0] = 0           // No stock on day 0
    dp[0][1] = -prices[0]  // Buy stock on day 0
    
    for i := 1; i < n; i++ {
        // No stock: either keep no stock or sell today
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
        
        // Have stock: either keep stock or buy today
        dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
    }
    
    return dp[n-1][0]
}
```

#### Digit DP

```go
func CountNumbersWithUniqueDigits(n int) int {
    if n == 0 {
        return 1
    }
    
    // dp[i] = count of numbers with i unique digits
    dp := make([]int, n+1)
    dp[0] = 1
    dp[1] = 9
    
    for i := 2; i <= n; i++ {
        dp[i] = dp[i-1] * (10 - i + 1)
    }
    
    result := 0
    for i := 0; i <= n; i++ {
        result += dp[i]
    }
    
    return result
}
```

---

## 🔤 String Algorithms

### 1. Pattern Matching

#### KMP Algorithm

```go
func KMPSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    // Build failure function
    failure := make([]int, m)
    j := 0
    for i := 1; i < m; i++ {
        for j > 0 && pattern[i] != pattern[j] {
            j = failure[j-1]
        }
        if pattern[i] == pattern[j] {
            j++
        }
        failure[i] = j
    }
    
    // Search
    result := []int{}
    j = 0
    for i := 0; i < n; i++ {
        for j > 0 && text[i] != pattern[j] {
            j = failure[j-1]
        }
        if text[i] == pattern[j] {
            j++
        }
        if j == m {
            result = append(result, i-m+1)
            j = failure[j-1]
        }
    }
    
    return result
}
```

#### Rabin-Karp Algorithm

```go
func RabinKarpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || m > n {
        return []int{}
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
    power := 1
    for i := 0; i < m; i++ {
        textHash = (textHash*base + int(text[i])) % mod
        if i < m-1 {
            power = (power * base) % mod
        }
    }
    
    result := []int{}
    
    // Check first window
    if textHash == patternHash && text[:m] == pattern {
        result = append(result, 0)
    }
    
    // Slide the window
    for i := m; i < n; i++ {
        // Remove leading character
        textHash = (textHash - int(text[i-m])*power) % mod
        if textHash < 0 {
            textHash += mod
        }
        
        // Add trailing character
        textHash = (textHash*base + int(text[i])) % mod
        
        // Check if hash matches and strings are equal
        if textHash == patternHash && text[i-m+1:i+1] == pattern {
            result = append(result, i-m+1)
        }
    }
    
    return result
}
```

### 2. String Processing

#### Suffix Array

```go
func BuildSuffixArray(text string) []int {
    n := len(text)
    suffixes := make([]Suffix, n)
    
    for i := 0; i < n; i++ {
        suffixes[i] = Suffix{
            index: i,
            rank:  [2]int{int(text[i]), 0},
        }
    }
    
    // Sort by first character
    sort.Slice(suffixes, func(i, j int) bool {
        return suffixes[i].rank[0] < suffixes[j].rank[0]
    })
    
    // Sort by first 2 characters, then 4, 8, etc.
    for k := 1; k < n; k *= 2 {
        // Assign new ranks
        prevRank := suffixes[0].rank[0]
        suffixes[0].rank[0] = 0
        newIndex := 0
        
        for i := 1; i < n; i++ {
            if suffixes[i].rank[0] == prevRank && 
               suffixes[i].rank[1] == suffixes[i-1].rank[1] {
                suffixes[i].rank[0] = newIndex
            } else {
                prevRank = suffixes[i].rank[0]
                newIndex++
                suffixes[i].rank[0] = newIndex
            }
        }
        
        // Assign second rank
        for i := 0; i < n; i++ {
            nextIndex := suffixes[i].index + k
            if nextIndex < n {
                suffixes[i].rank[1] = suffixes[getSuffixIndex(suffixes, nextIndex)].rank[0]
            } else {
                suffixes[i].rank[1] = -1
            }
        }
        
        // Sort by both ranks
        sort.Slice(suffixes, func(i, j int) bool {
            if suffixes[i].rank[0] != suffixes[j].rank[0] {
                return suffixes[i].rank[0] < suffixes[j].rank[0]
            }
            return suffixes[i].rank[1] < suffixes[j].rank[1]
        })
    }
    
    result := make([]int, n)
    for i := 0; i < n; i++ {
        result[i] = suffixes[i].index
    }
    
    return result
}

type Suffix struct {
    index int
    rank  [2]int
}
```

#### Longest Common Prefix

```go
func BuildLCPArray(text string, suffixArray []int) []int {
    n := len(text)
    lcp := make([]int, n)
    invSuffix := make([]int, n)
    
    for i := 0; i < n; i++ {
        invSuffix[suffixArray[i]] = i
    }
    
    k := 0
    for i := 0; i < n; i++ {
        if invSuffix[i] == n-1 {
            k = 0
            continue
        }
        
        j := suffixArray[invSuffix[i]+1]
        
        for i+k < n && j+k < n && text[i+k] == text[j+k] {
            k++
        }
        
        lcp[invSuffix[i]] = k
        
        if k > 0 {
            k--
        }
    }
    
    return lcp
}
```

---

## 🔢 Mathematical Algorithms

### 1. Number Theory

#### Greatest Common Divisor

```go
func GCD(a, b int) int {
    for b != 0 {
        a, b = b, a%b
    }
    return a
}

func ExtendedGCD(a, b int) (int, int, int) {
    if a == 0 {
        return b, 0, 1
    }
    
    gcd, x1, y1 := ExtendedGCD(b%a, a)
    x := y1 - (b/a)*x1
    y := x1
    
    return gcd, x, y
}
```

#### Modular Exponentiation

```go
func ModPow(base, exp, mod int) int {
    result := 1
    base %= mod
    
    for exp > 0 {
        if exp%2 == 1 {
            result = (result * base) % mod
        }
        exp >>= 1
        base = (base * base) % mod
    }
    
    return result
}
```

#### Prime Number Generation

```go
func SieveOfEratosthenes(n int) []bool {
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
    
    return isPrime
}

func GeneratePrimes(n int) []int {
    isPrime := SieveOfEratosthenes(n)
    primes := []int{}
    
    for i := 2; i <= n; i++ {
        if isPrime[i] {
            primes = append(primes, i)
        }
    }
    
    return primes
}
```

### 2. Combinatorics

#### Binomial Coefficient

```go
func BinomialCoefficient(n, k int) int {
    if k > n-k {
        k = n - k
    }
    
    result := 1
    for i := 0; i < k; i++ {
        result = result * (n - i) / (i + 1)
    }
    
    return result
}

func BinomialCoefficientMod(n, k, mod int) int {
    if k > n-k {
        k = n - k
    }
    
    numerator := 1
    for i := 0; i < k; i++ {
        numerator = (numerator * (n - i)) % mod
    }
    
    denominator := 1
    for i := 1; i <= k; i++ {
        denominator = (denominator * i) % mod
    }
    
    return (numerator * ModInverse(denominator, mod)) % mod
}

func ModInverse(a, mod int) int {
    _, x, _ := ExtendedGCD(a, mod)
    return (x%mod + mod) % mod
}
```

#### Permutations and Combinations

```go
func Permutations(n, r int) int {
    result := 1
    for i := 0; i < r; i++ {
        result *= (n - i)
    }
    return result
}

func Combinations(n, r int) int {
    if r > n-r {
        r = n - r
    }
    return BinomialCoefficient(n, r)
}

func GeneratePermutations(elements []int) [][]int {
    if len(elements) == 0 {
        return [][]int{{}}
    }
    
    result := [][]int{}
    for i, element := range elements {
        remaining := make([]int, len(elements)-1)
        copy(remaining, elements[:i])
        copy(remaining[i:], elements[i+1:])
        
        for _, perm := range GeneratePermutations(remaining) {
            result = append(result, append([]int{element}, perm...))
        }
    }
    
    return result
}
```

---

## 📐 Computational Geometry

### 1. Basic Geometric Operations

#### Point and Vector Operations

```go
type Point struct {
    X, Y float64
}

type Vector struct {
    X, Y float64
}

func (p Point) DistanceTo(q Point) float64 {
    dx := p.X - q.X
    dy := p.Y - q.Y
    return math.Sqrt(dx*dx + dy*dy)
}

func (p Point) VectorTo(q Point) Vector {
    return Vector{X: q.X - p.X, Y: q.Y - p.Y}
}

func (v Vector) Dot(w Vector) float64 {
    return v.X*w.X + v.Y*w.Y
}

func (v Vector) Cross(w Vector) float64 {
    return v.X*w.Y - v.Y*w.X
}

func (v Vector) Magnitude() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
```

#### Convex Hull (Graham Scan)

```go
func ConvexHull(points []Point) []Point {
    n := len(points)
    if n < 3 {
        return points
    }
    
    // Find bottom-most point (and leftmost in case of tie)
    start := 0
    for i := 1; i < n; i++ {
        if points[i].Y < points[start].Y || 
           (points[i].Y == points[start].Y && points[i].X < points[start].X) {
            start = i
        }
    }
    
    // Swap start point to beginning
    points[0], points[start] = points[start], points[0]
    
    // Sort points by polar angle with respect to start point
    sort.Slice(points[1:], func(i, j int) bool {
        p1 := points[0].VectorTo(points[i+1])
        p2 := points[0].VectorTo(points[j+1])
        cross := p1.Cross(p2)
        if cross == 0 {
            return p1.Magnitude() < p2.Magnitude()
        }
        return cross > 0
    })
    
    // Build convex hull
    hull := []Point{points[0], points[1]}
    
    for i := 2; i < n; i++ {
        for len(hull) > 1 {
            p1 := hull[len(hull)-2]
            p2 := hull[len(hull)-1]
            p3 := points[i]
            
            v1 := p1.VectorTo(p2)
            v2 := p2.VectorTo(p3)
            
            if v1.Cross(v2) > 0 {
                break
            }
            hull = hull[:len(hull)-1]
        }
        hull = append(hull, points[i])
    }
    
    return hull
}
```

### 2. Line Intersection

```go
type Line struct {
    A, B, C float64 // Ax + By + C = 0
}

func LineFromPoints(p1, p2 Point) Line {
    A := p2.Y - p1.Y
    B := p1.X - p2.X
    C := p1.Y*(p2.X-p1.X) - p1.X*(p2.Y-p1.Y)
    return Line{A: A, B: B, C: C}
}

func (l1 Line) Intersection(l2 Line) (Point, bool) {
    denominator := l1.A*l2.B - l2.A*l1.B
    if math.Abs(denominator) < 1e-9 {
        return Point{}, false // Lines are parallel
    }
    
    x := (l1.B*l2.C - l2.B*l1.C) / denominator
    y := (l2.A*l1.C - l1.A*l2.C) / denominator
    
    return Point{X: x, Y: y}, true
}
```

---

## 🔍 Advanced Sorting & Searching

### 1. Advanced Sorting Algorithms

#### Quick Sort with Optimizations

```go
func QuickSort(arr []int) {
    if len(arr) <= 1 {
        return
    }
    
    // Use insertion sort for small arrays
    if len(arr) <= 10 {
        InsertionSort(arr)
        return
    }
    
    // Choose pivot using median-of-three
    pivot := medianOfThree(arr, 0, len(arr)/2, len(arr)-1)
    arr[0], arr[pivot] = arr[pivot], arr[0]
    
    // Partition
    pivotIndex := partition(arr)
    
    // Recursively sort subarrays
    QuickSort(arr[:pivotIndex])
    QuickSort(arr[pivotIndex+1:])
}

func medianOfThree(arr []int, a, b, c int) int {
    if (arr[a] <= arr[b] && arr[b] <= arr[c]) || (arr[c] <= arr[b] && arr[b] <= arr[a]) {
        return b
    } else if (arr[b] <= arr[a] && arr[a] <= arr[c]) || (arr[c] <= arr[a] && arr[a] <= arr[b]) {
        return a
    } else {
        return c
    }
}

func partition(arr []int) int {
    pivot := arr[0]
    i, j := 1, len(arr)-1
    
    for i <= j {
        for i <= j && arr[i] <= pivot {
            i++
        }
        for i <= j && arr[j] > pivot {
            j--
        }
        if i < j {
            arr[i], arr[j] = arr[j], arr[i]
        }
    }
    
    arr[0], arr[j] = arr[j], arr[0]
    return j
}
```

#### Tim Sort (Stable Merge Sort)

```go
const MIN_MERGE = 32

func TimSort(arr []int) {
    n := len(arr)
    if n < 2 {
        return
    }
    
    // Sort individual runs of size MIN_MERGE
    for i := 0; i < n; i += MIN_MERGE {
        end := min(i+MIN_MERGE, n)
        InsertionSort(arr[i:end])
    }
    
    // Merge runs
    for size := MIN_MERGE; size < n; size *= 2 {
        for left := 0; left < n; left += 2 * size {
            mid := min(left+size, n)
            right := min(left+2*size, n)
            merge(arr, left, mid, right)
        }
    }
}

func merge(arr []int, left, mid, right int) {
    leftArr := make([]int, mid-left)
    rightArr := make([]int, right-mid)
    
    copy(leftArr, arr[left:mid])
    copy(rightArr, arr[mid:right])
    
    i, j, k := 0, 0, left
    
    for i < len(leftArr) && j < len(rightArr) {
        if leftArr[i] <= rightArr[j] {
            arr[k] = leftArr[i]
            i++
        } else {
            arr[k] = rightArr[j]
            j++
        }
        k++
    }
    
    for i < len(leftArr) {
        arr[k] = leftArr[i]
        i++
        k++
    }
    
    for j < len(rightArr) {
        arr[k] = rightArr[j]
        j++
        k++
    }
}
```

### 2. Advanced Searching

#### Binary Search Variations

```go
// Find first occurrence
func FirstOccurrence(arr []int, target int) int {
    left, right := 0, len(arr)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            result = mid
            right = mid - 1
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

// Find last occurrence
func LastOccurrence(arr []int, target int) int {
    left, right := 0, len(arr)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            result = mid
            left = mid + 1
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

// Find insertion position
func InsertPosition(arr []int, target int) int {
    left, right := 0, len(arr)
    
    for left < right {
        mid := left + (right-left)/2
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}
```

---

## ⚡ Performance Analysis

### 1. Time Complexity Analysis

```go
// O(1) - Constant time
func GetFirstElement(arr []int) int {
    if len(arr) > 0 {
        return arr[0]
    }
    return -1
}

// O(log n) - Logarithmic time
func BinarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

// O(n) - Linear time
func LinearSearch(arr []int, target int) int {
    for i, val := range arr {
        if val == target {
            return i
        }
    }
    return -1
}

// O(n log n) - Linearithmic time
func MergeSort(arr []int) {
    if len(arr) <= 1 {
        return
    }
    
    mid := len(arr) / 2
    left := make([]int, mid)
    right := make([]int, len(arr)-mid)
    
    copy(left, arr[:mid])
    copy(right, arr[mid:])
    
    MergeSort(left)
    MergeSort(right)
    merge(arr, left, right)
}

// O(n²) - Quadratic time
func BubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}
```

### 2. Space Complexity Analysis

```go
// O(1) - Constant space
func Swap(a, b *int) {
    *a, *b = *b, *a
}

// O(n) - Linear space
func CopyArray(arr []int) []int {
    result := make([]int, len(arr))
    copy(result, arr)
    return result
}

// O(log n) - Logarithmic space (recursion stack)
func BinarySearchRecursive(arr []int, target, left, right int) int {
    if left > right {
        return -1
    }
    
    mid := left + (right-left)/2
    if arr[mid] == target {
        return mid
    } else if arr[mid] < target {
        return BinarySearchRecursive(arr, target, mid+1, right)
    } else {
        return BinarySearchRecursive(arr, target, left, mid-1)
    }
}
```

### 3. Algorithm Optimization Techniques

#### Memoization

```go
func FibonacciMemo(n int) int {
    memo := make(map[int]int)
    return fibonacciMemo(n, memo)
}

func fibonacciMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    result := fibonacciMemo(n-1, memo) + fibonacciMemo(n-2, memo)
    memo[n] = result
    return result
}
```

#### Tabulation

```go
func FibonacciTab(n int) int {
    if n <= 1 {
        return n
    }
    
    dp := make([]int, n+1)
    dp[0], dp[1] = 0, 1
    
    for i := 2; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    
    return dp[n]
}
```

#### Space Optimization

```go
func FibonacciOptimized(n int) int {
    if n <= 1 {
        return n
    }
    
    prev2, prev1 := 0, 1
    
    for i := 2; i <= n; i++ {
        current := prev1 + prev2
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}
```

---

## 🎯 Practice Problems

### Easy Problems
1. **Two Sum** - Hash map approach
2. **Valid Parentheses** - Stack approach
3. **Maximum Subarray** - Kadane's algorithm
4. **Climbing Stairs** - Fibonacci pattern
5. **Best Time to Buy and Sell Stock** - Greedy approach

### Medium Problems
1. **Longest Palindromic Substring** - Expand around centers
2. **3Sum** - Two pointers technique
3. **Longest Substring Without Repeating Characters** - Sliding window
4. **Product of Array Except Self** - Prefix and suffix products
5. **Spiral Matrix** - Simulation approach

### Hard Problems
1. **Median of Two Sorted Arrays** - Binary search
2. **Regular Expression Matching** - Dynamic programming
3. **Merge k Sorted Lists** - Divide and conquer
4. **Trapping Rain Water** - Two pointers
5. **Word Ladder** - BFS with optimization

---

## 📚 Additional Resources

### Books
- **"Introduction to Algorithms"** by Cormen, Leiserson, Rivest, and Stein
- **"Algorithm Design Manual"** by Steven Skiena
- **"Programming Pearls"** by Jon Bentley
- **"Elements of Programming Interviews"** by Aziz, Lee, and Prakash

### Online Platforms
- **LeetCode**: 2000+ problems with solutions
- **HackerRank**: Algorithm challenges
- **Codeforces**: Competitive programming
- **AtCoder**: Japanese competitive programming

### Practice Strategies
1. **Start with basics**: Master fundamental data structures
2. **Pattern recognition**: Learn common algorithm patterns
3. **Time management**: Practice under time constraints
4. **Code quality**: Write clean, readable code
5. **Edge cases**: Always consider boundary conditions

---

**🚀 Master these advanced algorithms and you'll be ready for any technical interview! Practice regularly and focus on understanding the underlying principles. Good luck! 🎯**


##  Concurrent Algorithms

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-concurrent-algorithms -->

Placeholder content. Please replace with proper section.


##  Algorithm Design Patterns

<!-- AUTO-GENERATED ANCHOR: originally referenced as #-algorithm-design-patterns -->

Placeholder content. Please replace with proper section.
