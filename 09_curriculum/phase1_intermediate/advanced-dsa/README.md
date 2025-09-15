# Advanced Data Structures & Algorithms

## Table of Contents

1. [Overview](#overview)
2. [Dynamic Programming Advanced](#dynamic-programming-advanced)
3. [Graph Algorithms](#graph-algorithms)
4. [Tree Structures](#tree-structures)
5. [Advanced Sorting](#advanced-sorting)
6. [String Algorithms](#string-algorithms)
7. [Mathematical Algorithms](#mathematical-algorithms)
8. [Applications](#applications)
9. [Implementations](#implementations)
10. [Follow-up Questions](#follow-up-questions)
11. [Sources](#sources)
12. [Projects](#projects)

## Overview

### Learning Objectives

- Master advanced dynamic programming patterns
- Implement complex graph algorithms
- Understand advanced tree structures
- Apply optimization techniques
- Solve complex algorithmic problems

### What is Advanced DSA?

Advanced Data Structures & Algorithms builds upon fundamental concepts to solve complex problems efficiently. This module covers optimization techniques, advanced patterns, and real-world applications.

## Dynamic Programming Advanced

### 1. Advanced DP Patterns

#### Longest Common Subsequence (LCS)
```go
package main

import "fmt"

func longestCommonSubsequence(text1, text2 string) int {
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

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    text1 := "abcde"
    text2 := "ace"
    result := longestCommonSubsequence(text1, text2)
    fmt.Printf("LCS length: %d\n", result) // 3
}
```

#### Edit Distance (Levenshtein)
```go
package main

import "fmt"

func minDistance(word1, word2 string) int {
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
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
    }
    
    return dp[m][n]
}

func min(a, b, c int) int {
    if a < b && a < c {
        return a
    }
    if b < c {
        return b
    }
    return c
}

func main() {
    word1 := "horse"
    word2 := "ros"
    result := minDistance(word1, word2)
    fmt.Printf("Edit distance: %d\n", result) // 3
}
```

### 2. State Space Reduction

#### House Robber with Circular Array
```go
package main

import "fmt"

func rob(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    if len(nums) == 1 {
        return nums[0]
    }
    
    // Two cases: rob first house or don't rob first house
    return max(robLinear(nums[1:]), robLinear(nums[:len(nums)-1]))
}

func robLinear(nums []int) int {
    prev2, prev1 := 0, 0
    
    for _, num := range nums {
        current := max(prev1, prev2+num)
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    nums := []int{2, 3, 2}
    result := rob(nums)
    fmt.Printf("Maximum money: %d\n", result) // 3
}
```

## Graph Algorithms

### 1. Shortest Path Algorithms

#### Dijkstra's Algorithm
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
    adj [][]Edge
    n   int
}

func NewGraph(n int) *Graph {
    return &Graph{
        adj: make([][]Edge, n),
        n:   n,
    }
}

func (g *Graph) AddEdge(from, to, weight int) {
    g.adj[from] = append(g.adj[from], Edge{to, weight})
}

type Item struct {
    node     int
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

func (g *Graph) Dijkstra(start int) []int {
    dist := make([]int, g.n)
    for i := range dist {
        dist[i] = math.MaxInt32
    }
    dist[start] = 0
    
    pq := make(PriorityQueue, 0)
    heap.Init(&pq)
    heap.Push(&pq, &Item{start, 0, 0})
    
    for pq.Len() > 0 {
        item := heap.Pop(&pq).(*Item)
        u := item.node
        
        if item.distance > dist[u] {
            continue
        }
        
        for _, edge := range g.adj[u] {
            v := edge.to
            w := edge.weight
            
            if dist[u]+w < dist[v] {
                dist[v] = dist[u] + w
                heap.Push(&pq, &Item{v, dist[v], 0})
            }
        }
    }
    
    return dist
}

func main() {
    g := NewGraph(5)
    g.AddEdge(0, 1, 4)
    g.AddEdge(0, 2, 1)
    g.AddEdge(1, 2, 2)
    g.AddEdge(1, 3, 1)
    g.AddEdge(2, 3, 5)
    g.AddEdge(3, 4, 3)
    
    dist := g.Dijkstra(0)
    fmt.Println("Shortest distances from node 0:", dist)
}
```

#### Floyd-Warshall Algorithm
```go
package main

import (
    "fmt"
    "math"
)

func floydWarshall(graph [][]int) [][]int {
    n := len(graph)
    dist := make([][]int, n)
    
    // Initialize distance matrix
    for i := 0; i < n; i++ {
        dist[i] = make([]int, n)
        for j := 0; j < n; j++ {
            if i == j {
                dist[i][j] = 0
            } else if graph[i][j] != 0 {
                dist[i][j] = graph[i][j]
            } else {
                dist[i][j] = math.MaxInt32
            }
        }
    }
    
    // Floyd-Warshall algorithm
    for k := 0; k < n; k++ {
        for i := 0; i < n; i++ {
            for j := 0; j < n; j++ {
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

func main() {
    graph := [][]int{
        {0, 5, 0, 10},
        {0, 0, 3, 0},
        {0, 0, 0, 1},
        {0, 0, 0, 0},
    }
    
    dist := floydWarshall(graph)
    fmt.Println("All-pairs shortest paths:")
    for i := 0; i < len(dist); i++ {
        fmt.Println(dist[i])
    }
}
```

### 2. Network Flow

#### Maximum Flow (Ford-Fulkerson)
```go
package main

import "fmt"

type Graph struct {
    capacity [][]int
    flow     [][]int
    n        int
}

func NewGraph(n int) *Graph {
    return &Graph{
        capacity: make([][]int, n),
        flow:     make([][]int, n),
        n:        n,
    }
}

func (g *Graph) AddEdge(from, to, capacity int) {
    g.capacity[from] = make([]int, g.n)
    g.flow[from] = make([]int, g.n)
    g.capacity[from][to] = capacity
}

func (g *Graph) maxFlow(source, sink int) int {
    parent := make([]int, g.n)
    maxFlow := 0
    
    for {
        // BFS to find augmenting path
        visited := make([]bool, g.n)
        queue := []int{source}
        visited[source] = true
        parent[source] = -1
        
        found := false
        for len(queue) > 0 && !found {
            u := queue[0]
            queue = queue[1:]
            
            for v := 0; v < g.n; v++ {
                if !visited[v] && g.capacity[u][v] > g.flow[u][v] {
                    parent[v] = u
                    visited[v] = true
                    queue = append(queue, v)
                    
                    if v == sink {
                        found = true
                        break
                    }
                }
            }
        }
        
        if !found {
            break
        }
        
        // Find minimum residual capacity
        pathFlow := int(^uint(0) >> 1)
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            pathFlow = min(pathFlow, g.capacity[u][v]-g.flow[u][v])
        }
        
        // Update flow
        for v := sink; v != source; v = parent[v] {
            u := parent[v]
            g.flow[u][v] += pathFlow
            g.flow[v][u] -= pathFlow
        }
        
        maxFlow += pathFlow
    }
    
    return maxFlow
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func main() {
    g := NewGraph(6)
    g.AddEdge(0, 1, 16)
    g.AddEdge(0, 2, 13)
    g.AddEdge(1, 2, 10)
    g.AddEdge(1, 3, 12)
    g.AddEdge(2, 1, 4)
    g.AddEdge(2, 4, 14)
    g.AddEdge(3, 2, 9)
    g.AddEdge(3, 5, 20)
    g.AddEdge(4, 3, 7)
    g.AddEdge(4, 5, 4)
    
    maxFlow := g.maxFlow(0, 5)
    fmt.Printf("Maximum flow: %d\n", maxFlow)
}
```

## Tree Structures

### 1. Advanced Tree Operations

#### AVL Tree Implementation
```go
package main

import "fmt"

type AVLNode struct {
    key    int
    height int
    left   *AVLNode
    right  *AVLNode
}

type AVLTree struct {
    root *AVLNode
}

func NewAVLTree() *AVLTree {
    return &AVLTree{}
}

func (n *AVLNode) getHeight() int {
    if n == nil {
        return 0
    }
    return n.height
}

func (n *AVLNode) getBalance() int {
    if n == nil {
        return 0
    }
    return n.left.getHeight() - n.right.getHeight()
}

func (n *AVLNode) updateHeight() {
    n.height = 1 + max(n.left.getHeight(), n.right.getHeight())
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func (t *AVLTree) rightRotate(y *AVLNode) *AVLNode {
    x := y.left
    T2 := x.right
    
    x.right = y
    y.left = T2
    
    y.updateHeight()
    x.updateHeight()
    
    return x
}

func (t *AVLTree) leftRotate(x *AVLNode) *AVLNode {
    y := x.right
    T2 := y.left
    
    y.left = x
    x.right = T2
    
    x.updateHeight()
    y.updateHeight()
    
    return y
}

func (t *AVLTree) insert(node *AVLNode, key int) *AVLNode {
    if node == nil {
        return &AVLNode{key: key, height: 1}
    }
    
    if key < node.key {
        node.left = t.insert(node.left, key)
    } else if key > node.key {
        node.right = t.insert(node.right, key)
    } else {
        return node
    }
    
    node.updateHeight()
    
    balance := node.getBalance()
    
    // Left Left Case
    if balance > 1 && key < node.left.key {
        return t.rightRotate(node)
    }
    
    // Right Right Case
    if balance < -1 && key > node.right.key {
        return t.leftRotate(node)
    }
    
    // Left Right Case
    if balance > 1 && key > node.left.key {
        node.left = t.leftRotate(node.left)
        return t.rightRotate(node)
    }
    
    // Right Left Case
    if balance < -1 && key < node.right.key {
        node.right = t.rightRotate(node.right)
        return t.leftRotate(node)
    }
    
    return node
}

func (t *AVLTree) Insert(key int) {
    t.root = t.insert(t.root, key)
}

func (t *AVLTree) inorder(node *AVLNode) {
    if node != nil {
        t.inorder(node.left)
        fmt.Printf("%d ", node.key)
        t.inorder(node.right)
    }
}

func (t *AVLTree) Inorder() {
    t.inorder(t.root)
    fmt.Println()
}

func main() {
    tree := NewAVLTree()
    
    keys := []int{10, 20, 30, 40, 50, 25}
    for _, key := range keys {
        tree.Insert(key)
    }
    
    fmt.Println("Inorder traversal of AVL tree:")
    tree.Inorder()
}
```

### 2. Segment Tree

#### Range Sum Query
```go
package main

import "fmt"

type SegmentTree struct {
    tree []int
    n    int
}

func NewSegmentTree(arr []int) *SegmentTree {
    n := len(arr)
    st := &SegmentTree{
        tree: make([]int, 4*n),
        n:    n,
    }
    st.build(arr, 0, 0, n-1)
    return st
}

func (st *SegmentTree) build(arr []int, node, start, end int) {
    if start == end {
        st.tree[node] = arr[start]
    } else {
        mid := (start + end) / 2
        st.build(arr, 2*node+1, start, mid)
        st.build(arr, 2*node+2, mid+1, end)
        st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
    }
}

func (st *SegmentTree) query(node, start, end, l, r int) int {
    if r < start || end < l {
        return 0
    }
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    mid := (start + end) / 2
    leftSum := st.query(2*node+1, start, mid, l, r)
    rightSum := st.query(2*node+2, mid+1, end, l, r)
    
    return leftSum + rightSum
}

func (st *SegmentTree) Query(l, r int) int {
    return st.query(0, 0, st.n-1, l, r)
}

func (st *SegmentTree) update(node, start, end, idx, val int) {
    if start == end {
        st.tree[node] = val
    } else {
        mid := (start + end) / 2
        if idx <= mid {
            st.update(2*node+1, start, mid, idx, val)
        } else {
            st.update(2*node+2, mid+1, end, idx, val)
        }
        st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
    }
}

func (st *SegmentTree) Update(idx, val int) {
    st.update(0, 0, st.n-1, idx, val)
}

func main() {
    arr := []int{1, 3, 5, 7, 9, 11}
    st := NewSegmentTree(arr)
    
    fmt.Printf("Sum of range [1, 3]: %d\n", st.Query(1, 3)) // 15
    fmt.Printf("Sum of range [0, 5]: %d\n", st.Query(0, 5)) // 36
    
    st.Update(1, 10)
    fmt.Printf("Sum of range [1, 3] after update: %d\n", st.Query(1, 3)) // 22
}
```

## Advanced Sorting

### 1. Heap Sort

```go
package main

import "fmt"

func heapSort(arr []int) {
    n := len(arr)
    
    // Build max heap
    for i := n/2 - 1; i >= 0; i-- {
        heapify(arr, n, i)
    }
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    }
}

func heapify(arr []int, n, i int) {
    largest := i
    left := 2*i + 1
    right := 2*i + 2
    
    if left < n && arr[left] > arr[largest] {
        largest = left
    }
    
    if right < n && arr[right] > arr[largest] {
        largest = right
    }
    
    if largest != i {
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
    }
}

func main() {
    arr := []int{64, 34, 25, 12, 22, 11, 90}
    fmt.Println("Original array:", arr)
    
    heapSort(arr)
    fmt.Println("Sorted array:", arr)
}
```

### 2. Quick Sort with Optimizations

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
)

func quickSort(arr []int) {
    if len(arr) < 2 {
        return
    }
    
    // Use random pivot for better average case
    rand.Seed(time.Now().UnixNano())
    pivot := rand.Intn(len(arr))
    arr[0], arr[pivot] = arr[pivot], arr[0]
    
    left, right := partition(arr)
    
    quickSort(arr[:left])
    quickSort(arr[right:])
}

func partition(arr []int) (int, int) {
    pivot := arr[0]
    left := 0
    right := len(arr)
    
    for i := 1; i < right; {
        if arr[i] < pivot {
            left++
            arr[i], arr[left] = arr[left], arr[i]
            i++
        } else if arr[i] > pivot {
            right--
            arr[i], arr[right] = arr[right], arr[i]
        } else {
            i++
        }
    }
    
    arr[0], arr[left] = arr[left], arr[0]
    return left, right
}

func main() {
    arr := []int{64, 34, 25, 12, 22, 11, 90}
    fmt.Println("Original array:", arr)
    
    quickSort(arr)
    fmt.Println("Sorted array:", arr)
}
```

## String Algorithms

### 1. KMP Algorithm

```go
package main

import "fmt"

func kmpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 {
        return []int{}
    }
    
    lps := computeLPS(pattern)
    result := []int{}
    
    i, j := 0, 0
    for i < n {
        if text[i] == pattern[j] {
            i++
            j++
        }
        
        if j == m {
            result = append(result, i-j)
            j = lps[j-1]
        } else if i < n && text[i] != pattern[j] {
            if j != 0 {
                j = lps[j-1]
            } else {
                i++
            }
        }
    }
    
    return result
}

func computeLPS(pattern string) []int {
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

func main() {
    text := "ABABDABACDABABCABAB"
    pattern := "ABABCABAB"
    
    result := kmpSearch(text, pattern)
    fmt.Printf("Pattern found at indices: %v\n", result)
}
```

### 2. Rabin-Karp Algorithm

```go
package main

import "fmt"

func rabinKarpSearch(text, pattern string) []int {
    n, m := len(text), len(pattern)
    if m == 0 || m > n {
        return []int{}
    }
    
    const base = 256
    const mod = 101
    
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
    
    result := []int{}
    
    // Check first window
    if patternHash == textHash && text[:m] == pattern {
        result = append(result, 0)
    }
    
    // Calculate base^(m-1) for rolling hash
    h := 1
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    // Rolling hash
    for i := 1; i <= n-m; i++ {
        // Remove leading digit, add trailing digit
        textHash = (textHash - int(text[i-1])*h + mod) % mod
        textHash = (textHash*base + int(text[i+m-1])) % mod
        
        if patternHash == textHash && text[i:i+m] == pattern {
            result = append(result, i)
        }
    }
    
    return result
}

func main() {
    text := "ABABDABACDABABCABAB"
    pattern := "ABABCABAB"
    
    result := rabinKarpSearch(text, pattern)
    fmt.Printf("Pattern found at indices: %v\n", result)
}
```

## Mathematical Algorithms

### 1. Fast Exponentiation

```go
package main

import "fmt"

func fastExponentiation(base, exponent, mod int) int {
    result := 1
    base = base % mod
    
    for exponent > 0 {
        if exponent%2 == 1 {
            result = (result * base) % mod
        }
        exponent = exponent >> 1
        base = (base * base) % mod
    }
    
    return result
}

func main() {
    base := 2
    exponent := 1000000
    mod := 1000000007
    
    result := fastExponentiation(base, exponent, mod)
    fmt.Printf("%d^%d mod %d = %d\n", base, exponent, mod, result)
}
```

### 2. Extended Euclidean Algorithm

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

func modularInverse(a, m int) int {
    gcd, x, _ := extendedGCD(a, m)
    if gcd != 1 {
        return -1 // Inverse doesn't exist
    }
    
    return ((x % m) + m) % m
}

func main() {
    a, b := 56, 15
    gcd, x, y := extendedGCD(a, b)
    fmt.Printf("GCD(%d, %d) = %d\n", a, b, gcd)
    fmt.Printf("x = %d, y = %d\n", x, y)
    fmt.Printf("%d*%d + %d*%d = %d\n", a, x, b, y, a*x+b*y)
    
    // Modular inverse
    a, m := 3, 11
    inv := modularInverse(a, m)
    if inv != -1 {
        fmt.Printf("Modular inverse of %d mod %d is %d\n", a, m, inv)
    } else {
        fmt.Printf("Modular inverse of %d mod %d doesn't exist\n", a, m)
    }
}
```

## Follow-up Questions

### 1. Algorithm Design
**Q: How do you choose between different graph algorithms?**
A: Consider the problem constraints: use BFS for unweighted shortest paths, Dijkstra for weighted non-negative edges, Bellman-Ford for negative weights, Floyd-Warshall for all-pairs shortest paths.

### 2. Optimization Techniques
**Q: When should you use memoization vs tabulation in DP?**
A: Use memoization for problems with sparse state space or when you don't need all subproblems. Use tabulation for problems where you need all subproblems or want to optimize space.

### 3. Data Structure Selection
**Q: When would you use a segment tree vs a binary indexed tree?**
A: Use segment trees for range queries and updates, especially when you need complex range operations. Use BIT for prefix sum queries and point updates, as it's simpler and more memory efficient.

## Sources

### Books
- **Introduction to Algorithms** by CLRS
- **Algorithm Design Manual** by Steven Skiena
- **Competitive Programming** by Steven Halim

### Online Resources
- **LeetCode** - Advanced algorithm problems
- **Codeforces** - Competitive programming
- **TopCoder** - Algorithm tutorials

## Projects

### 1. Algorithm Visualizer
**Objective**: Build a tool to visualize advanced algorithms
**Requirements**: Graph algorithms, sorting, dynamic programming
**Deliverables**: Interactive algorithm visualization tool

### 2. Competitive Programming Library
**Objective**: Create a library of advanced algorithms
**Requirements**: All major algorithm categories, optimized implementations
**Deliverables**: Complete algorithm library with documentation

### 3. Performance Benchmarking Tool
**Objective**: Build a tool to compare algorithm performance
**Requirements**: Multiple algorithms, timing, memory usage
**Deliverables**: Comprehensive benchmarking tool

---

**Next**: [Operating Systems Deep Dive](./os-deep-dive/README.md) | **Previous**: [Phase 1](../README.md) | **Up**: [Phase 1](../README.md)

