# 🚀 Google Advanced Algorithms & Data Structures Guide

> **Master the advanced algorithms and data structures that Google interviews emphasize**

## 📋 Table of Contents

1. [🔗 Advanced Graph Algorithms](#-advanced-graph-algorithms)
2. [📊 Advanced Data Structures](#-advanced-data-structures)
3. [⚡ Dynamic Programming Patterns](#-dynamic-programming-patterns)
4. [🎯 String Algorithms](#-string-algorithms)
5. [🔢 Number Theory & Math](#-number-theory--math)
6. [🌐 Network Flow Algorithms](#-network-flow-algorithms)
7. [🎯 Google-Specific Algorithm Questions](#-google-specific-algorithm-questions)

---

## 🔗 Advanced Graph Algorithms

### **1. A* Search Algorithm**

**Problem**: Find the shortest path in a weighted graph with heuristics.

```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

type Node struct {
    ID       string
    X        float64
    Y        float64
    Neighbors map[string]float64
}

type AStarNode struct {
    Node     *Node
    G        float64
    H        float64
    F        float64
    Parent   *AStarNode
    Index    int
}

type PriorityQueue []*AStarNode

func (pq PriorityQueue) Len() int { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].F < pq[j].F }
func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
    pq[i].Index = i
    pq[j].Index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
    n := len(*pq)
    item := x.(*AStarNode)
    item.Index = n
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

type AStarGraph struct {
    nodes map[string]*Node
}

func NewAStarGraph() *AStarGraph {
    return &AStarGraph{nodes: make(map[string]*Node)}
}

func (g *AStarGraph) AddNode(id string, x, y float64) {
    g.nodes[id] = &Node{
        ID:        id,
        X:         x,
        Y:         y,
        Neighbors: make(map[string]float64),
    }
}

func (g *AStarGraph) AddEdge(from, to string, distance float64) {
    if fromNode, exists := g.nodes[from]; exists {
        fromNode.Neighbors[to] = distance
    }
    if toNode, exists := g.nodes[to]; exists {
        toNode.Neighbors[from] = distance
    }
}

func (g *AStarGraph) Heuristic(from, to string) float64 {
    fromNode := g.nodes[from]
    toNode := g.nodes[to]
    dx := fromNode.X - toNode.X
    dy := fromNode.Y - toNode.Y
    return math.Sqrt(dx*dx + dy*dy)
}

func (g *AStarGraph) AStar(start, goal string) ([]string, float64) {
    if _, exists := g.nodes[start]; !exists {
        return nil, 0
    }
    if _, exists := g.nodes[goal]; !exists {
        return nil, 0
    }
    
    openSet := &PriorityQueue{}
    heap.Init(openSet)
    
    closedSet := make(map[string]bool)
    cameFrom := make(map[string]*AStarNode)
    
    startNode := &AStarNode{
        Node:   g.nodes[start],
        G:      0,
        H:      g.Heuristic(start, goal),
        F:      g.Heuristic(start, goal),
        Parent: nil,
    }
    
    heap.Push(openSet, startNode)
    cameFrom[start] = startNode
    
    for openSet.Len() > 0 {
        current := heap.Pop(openSet).(*AStarNode)
        
        if current.Node.ID == goal {
            path := []string{}
            totalCost := current.G
            
            for current != nil {
                path = append([]string{current.Node.ID}, path...)
                current = current.Parent
            }
            
            return path, totalCost
        }
        
        closedSet[current.Node.ID] = true
        
        for neighborID, distance := range current.Node.Neighbors {
            if closedSet[neighborID] {
                continue
            }
            
            tentativeG := current.G + distance
            
            if existingNode, exists := cameFrom[neighborID]; exists {
                if tentativeG >= existingNode.G {
                    continue
                }
            }
            
            neighborNode := g.nodes[neighborID]
            h := g.Heuristic(neighborID, goal)
            f := tentativeG + h
            
            neighbor := &AStarNode{
                Node:   neighborNode,
                G:      tentativeG,
                H:      h,
                F:      f,
                Parent: current,
            }
            
            cameFrom[neighborID] = neighbor
            heap.Push(openSet, neighbor)
        }
    }
    
    return nil, 0
}

func main() {
    graph := NewAStarGraph()
    
    graph.AddNode("A", 0, 0)
    graph.AddNode("B", 1, 0)
    graph.AddNode("C", 2, 0)
    graph.AddNode("D", 0, 1)
    graph.AddNode("E", 1, 1)
    graph.AddNode("F", 2, 1)
    graph.AddNode("G", 0, 2)
    graph.AddNode("H", 1, 2)
    graph.AddNode("I", 2, 2)
    
    graph.AddEdge("A", "B", 1.0)
    graph.AddEdge("A", "D", 1.0)
    graph.AddEdge("B", "C", 1.0)
    graph.AddEdge("B", "E", 1.0)
    graph.AddEdge("C", "F", 1.0)
    graph.AddEdge("D", "E", 1.0)
    graph.AddEdge("D", "G", 1.0)
    graph.AddEdge("E", "F", 1.0)
    graph.AddEdge("E", "H", 1.0)
    graph.AddEdge("F", "I", 1.0)
    graph.AddEdge("G", "H", 1.0)
    graph.AddEdge("H", "I", 1.0)
    
    path, cost := graph.AStar("A", "I")
    
    if path != nil {
        fmt.Printf("Path found: %v\n", path)
        fmt.Printf("Total cost: %.2f\n", cost)
    } else {
        fmt.Println("No path found")
    }
}
```

### **2. Maximum Flow (Ford-Fulkerson)**

**Problem**: Find maximum flow in a network.

```go
package main

import (
    "fmt"
    "math"
)

type Edge struct {
    From     int
    To       int
    Capacity int
    Flow     int
}

type FlowNetwork struct {
    vertices int
    edges    [][]Edge
    residual [][]int
}

func NewFlowNetwork(vertices int) *FlowNetwork {
    return &FlowNetwork{
        vertices: vertices,
        edges:    make([][]Edge, vertices),
        residual: make([][]int, vertices),
    }
}

func (fn *FlowNetwork) AddEdge(from, to, capacity int) {
    edge := Edge{From: from, To: to, Capacity: capacity, Flow: 0}
    fn.edges[from] = append(fn.edges[from], edge)
    
    if fn.residual[from] == nil {
        fn.residual[from] = make([]int, fn.vertices)
    }
    if fn.residual[to] == nil {
        fn.residual[to] = make([]int, fn.vertices)
    }
    
    fn.residual[from][to] = capacity
}

func (fn *FlowNetwork) BFS(source, sink int) ([]int, bool) {
    parent := make([]int, fn.vertices)
    visited := make([]bool, fn.vertices)
    queue := []int{source}
    
    for i := range parent {
        parent[i] = -1
    }
    
    visited[source] = true
    
    for len(queue) > 0 {
        u := queue[0]
        queue = queue[1:]
        
        for v := 0; v < fn.vertices; v++ {
            if !visited[v] && fn.residual[u][v] > 0 {
                parent[v] = u
                visited[v] = true
                queue = append(queue, v)
                
                if v == sink {
                    path := []int{}
                    for v != -1 {
                        path = append([]int{v}, path...)
                        v = parent[v]
                    }
                    return path, true
                }
            }
        }
    }
    
    return nil, false
}

func (fn *FlowNetwork) MaxFlow(source, sink int) int {
    maxFlow := 0
    
    for {
        path, found := fn.BFS(source, sink)
        if !found {
            break
        }
        
        minCapacity := math.MaxInt32
        for i := 0; i < len(path)-1; i++ {
            u := path[i]
            v := path[i+1]
            if fn.residual[u][v] < minCapacity {
                minCapacity = fn.residual[u][v]
            }
        }
        
        for i := 0; i < len(path)-1; i++ {
            u := path[i]
            v := path[i+1]
            fn.residual[u][v] -= minCapacity
            fn.residual[v][u] += minCapacity
        }
        
        maxFlow += minCapacity
    }
    
    return maxFlow
}

func main() {
    network := NewFlowNetwork(6)
    
    network.AddEdge(0, 1, 16)
    network.AddEdge(0, 2, 13)
    network.AddEdge(1, 2, 10)
    network.AddEdge(1, 3, 12)
    network.AddEdge(2, 1, 4)
    network.AddEdge(2, 4, 14)
    network.AddEdge(3, 2, 9)
    network.AddEdge(3, 5, 20)
    network.AddEdge(4, 3, 7)
    network.AddEdge(4, 5, 4)
    
    maxFlow := network.MaxFlow(0, 5)
    fmt.Printf("Maximum flow: %d\n", maxFlow)
}
```

---

## 📊 Advanced Data Structures

### **3. Suffix Array**

**Problem**: Efficient substring search and pattern matching.

```go
package main

import (
    "fmt"
    "sort"
    "strings"
)

type SuffixArray struct {
    text  string
    array []int
}

func NewSuffixArray(text string) *SuffixArray {
    sa := &SuffixArray{
        text:  text,
        array: make([]int, len(text)),
    }
    
    suffixes := make([]string, len(text))
    for i := 0; i < len(text); i++ {
        suffixes[i] = text[i:]
    }
    
    sort.Strings(suffixes)
    for i, suffix := range suffixes {
        sa.array[i] = len(text) - len(suffix)
    }
    
    return sa
}

func (sa *SuffixArray) Search(pattern string) []int {
    var result []int
    
    left := sa.binarySearchLeft(pattern)
    right := sa.binarySearchRight(pattern)
    
    if left <= right {
        for i := left; i <= right; i++ {
            result = append(result, sa.array[i])
        }
    }
    
    return result
}

func (sa *SuffixArray) binarySearchLeft(pattern string) int {
    left, right := 0, len(sa.array)-1
    result := -1
    
    for left <= right {
        mid := (left + right) / 2
        suffix := sa.text[sa.array[mid]:]
        
        if strings.HasPrefix(suffix, pattern) {
            result = mid
            right = mid - 1
        } else if suffix < pattern {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

func (sa *SuffixArray) binarySearchRight(pattern string) int {
    left, right := 0, len(sa.array)-1
    result := -1
    
    for left <= right {
        mid := (left + right) / 2
        suffix := sa.text[sa.array[mid]:]
        
        if strings.HasPrefix(suffix, pattern) {
            result = mid
            left = mid + 1
        } else if suffix < pattern {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

func (sa *SuffixArray) LongestCommonPrefix(i, j int) int {
    if i == j {
        return len(sa.text) - sa.array[i]
    }
    
    suffix1 := sa.text[sa.array[i]:]
    suffix2 := sa.text[sa.array[j]:]
    
    minLen := len(suffix1)
    if len(suffix2) < minLen {
        minLen = len(suffix2)
    }
    
    for k := 0; k < minLen; k++ {
        if suffix1[k] != suffix2[k] {
            return k
        }
    }
    
    return minLen
}

func (sa *SuffixArray) LongestRepeatedSubstring() string {
    maxLen := 0
    maxIndex := -1
    
    for i := 0; i < len(sa.array)-1; i++ {
        lcp := sa.LongestCommonPrefix(i, i+1)
        if lcp > maxLen {
            maxLen = lcp
            maxIndex = sa.array[i]
        }
    }
    
    if maxLen > 0 {
        return sa.text[maxIndex : maxIndex+maxLen]
    }
    
    return ""
}

func main() {
    text := "banana"
    sa := NewSuffixArray(text)
    
    fmt.Printf("Text: %s\n", text)
    fmt.Printf("Suffix Array: %v\n", sa.array)
    
    patterns := []string{"an", "na", "ban", "xyz"}
    for _, pattern := range patterns {
        positions := sa.Search(pattern)
        fmt.Printf("Pattern '%s' found at positions: %v\n", pattern, positions)
    }
    
    lrs := sa.LongestRepeatedSubstring()
    fmt.Printf("Longest repeated substring: '%s'\n", lrs)
}
```

### **4. Fenwick Tree (Binary Indexed Tree)**

**Problem**: Efficient prefix sums and range updates.

```go
package main

import "fmt"

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

func (ft *FenwickTree) Update(i, delta int) {
    i++
    
    for i <= ft.size {
        ft.tree[i] += delta
        i += i & (-i)
    }
}

func (ft *FenwickTree) Query(i int) int {
    i++
    
    sum := 0
    for i > 0 {
        sum += ft.tree[i]
        i -= i & (-i)
    }
    
    return sum
}

func (ft *FenwickTree) RangeQuery(left, right int) int {
    return ft.Query(right) - ft.Query(left-1)
}

func (ft *FenwickTree) GetValue(i int) int {
    return ft.RangeQuery(i, i)
}

func (ft *FenwickTree) SetValue(i, value int) {
    current := ft.GetValue(i)
    ft.Update(i, value-current)
}

func main() {
    ft := NewFenwickTree(8)
    
    values := []int{1, 3, 5, 7, 9, 11, 13, 15}
    for i, val := range values {
        ft.Update(i, val)
    }
    
    fmt.Println("Fenwick Tree Operations:")
    
    for i := 0; i < 8; i++ {
        fmt.Printf("Prefix sum [0, %d]: %d\n", i, ft.Query(i))
    }
    
    fmt.Printf("Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))
    fmt.Printf("Range sum [1, 7]: %d\n", ft.RangeQuery(1, 7))
    
    ft.Update(3, 10)
    fmt.Printf("After update: Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))
    
    ft.SetValue(4, 20)
    fmt.Printf("After set: Range sum [2, 5]: %d\n", ft.RangeQuery(2, 5))
}
```

---

## ⚡ Dynamic Programming Patterns

### **5. Longest Increasing Subsequence (LIS)**

**Problem**: Find the length of the longest increasing subsequence.

```go
package main

import "fmt"

func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    dp := make([]int, len(nums))
    maxLen := 1
    
    for i := range dp {
        dp[i] = 1
    }
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[i] > nums[j] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
        maxLen = max(maxLen, dp[i])
    }
    
    return maxLen
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// Optimized version using binary search
func lengthOfLISOptimized(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    tails := make([]int, 0)
    
    for _, num := range nums {
        pos := binarySearch(tails, num)
        if pos == len(tails) {
            tails = append(tails, num)
        } else {
            tails[pos] = num
        }
    }
    
    return len(tails)
}

func binarySearch(nums []int, target int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := (left + right) / 2
        if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return left
}

func main() {
    nums := []int{10, 9, 2, 5, 3, 7, 101, 18}
    
    fmt.Printf("LIS length (O(n²)): %d\n", lengthOfLIS(nums))
    fmt.Printf("LIS length (O(n log n)): %d\n", lengthOfLISOptimized(nums))
}
```

### **6. Edit Distance (Levenshtein Distance)**

**Problem**: Find minimum operations to transform one string to another.

```go
package main

import "fmt"

func minDistance(word1, word2 string) int {
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
                dp[i][j] = min(
                    dp[i-1][j]+1,    // delete
                    dp[i][j-1]+1,    // insert
                    dp[i-1][j-1]+1,  // replace
                )
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
    
    distance := minDistance(word1, word2)
    fmt.Printf("Edit distance between '%s' and '%s': %d\n", word1, word2, distance)
}
```

---

## 🎯 String Algorithms

### **7. KMP (Knuth-Morris-Pratt) Algorithm**

**Problem**: Efficient string pattern matching.

```go
package main

import "fmt"

func kmpSearch(text, pattern string) []int {
    if len(pattern) == 0 {
        return []int{0}
    }
    
    lps := computeLPS(pattern)
    var result []int
    
    i, j := 0, 0
    
    for i < len(text) {
        if text[i] == pattern[j] {
            i++
            j++
        }
        
        if j == len(pattern) {
            result = append(result, i-j)
            j = lps[j-1]
        } else if i < len(text) && text[i] != pattern[j] {
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
    lps := make([]int, len(pattern))
    length := 0
    i := 1
    
    for i < len(pattern) {
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
    
    positions := kmpSearch(text, pattern)
    fmt.Printf("Pattern '%s' found at positions: %v\n", pattern, positions)
}
```

### **8. Rabin-Karp Algorithm**

**Problem**: Pattern matching using rolling hash.

```go
package main

import "fmt"

const base = 256
const mod = 101

func rabinKarpSearch(text, pattern string) []int {
    if len(pattern) == 0 || len(pattern) > len(text) {
        return []int{}
    }
    
    var result []int
    n, m := len(text), len(pattern)
    
    // Calculate hash of pattern and first window of text
    patternHash := 0
    textHash := 0
    h := 1
    
    // Calculate h = base^(m-1) % mod
    for i := 0; i < m-1; i++ {
        h = (h * base) % mod
    }
    
    // Calculate initial hashes
    for i := 0; i < m; i++ {
        patternHash = (base*patternHash + int(pattern[i])) % mod
        textHash = (base*textHash + int(text[i])) % mod
    }
    
    // Slide the pattern over text
    for i := 0; i <= n-m; i++ {
        // Check if hashes match
        if patternHash == textHash {
            // Check characters one by one
            j := 0
            for j < m && text[i+j] == pattern[j] {
                j++
            }
            
            if j == m {
                result = append(result, i)
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
    
    return result
}

func main() {
    text := "GEEKS FOR GEEKS"
    pattern := "GEEK"
    
    positions := rabinKarpSearch(text, pattern)
    fmt.Printf("Pattern '%s' found at positions: %v\n", pattern, positions)
}
```

---

## 🎯 Google-Specific Algorithm Questions

### **9. Design a Rate Limiter**

**Problem**: Implement a rate limiter that allows N requests per second.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type RateLimiter struct {
    requests   []time.Time
    maxRequests int
    window     time.Duration
    mutex      sync.Mutex
}

func NewRateLimiter(maxRequests int, window time.Duration) *RateLimiter {
    return &RateLimiter{
        requests:    make([]time.Time, 0),
        maxRequests: maxRequests,
        window:      window,
    }
}

func (rl *RateLimiter) Allow() bool {
    rl.mutex.Lock()
    defer rl.mutex.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-rl.window)
    
    // Remove old requests
    var validRequests []time.Time
    for _, reqTime := range rl.requests {
        if reqTime.After(cutoff) {
            validRequests = append(validRequests, reqTime)
        }
    }
    rl.requests = validRequests
    
    // Check if under limit
    if len(rl.requests) < rl.maxRequests {
        rl.requests = append(rl.requests, now)
        return true
    }
    
    return false
}

func main() {
    limiter := NewRateLimiter(5, time.Second)
    
    for i := 0; i < 10; i++ {
        allowed := limiter.Allow()
        fmt.Printf("Request %d: %t\n", i+1, allowed)
        time.Sleep(100 * time.Millisecond)
    }
}
```

### **10. Design a Distributed Cache**

**Problem**: Implement a distributed cache with consistent hashing.

```go
package main

import (
    "crypto/md5"
    "fmt"
    "sort"
    "strconv"
    "sync"
)

type Node struct {
    ID   string
    Hash uint32
}

type ConsistentHash struct {
    nodes    []Node
    replicas int
    mutex    sync.RWMutex
}

func NewConsistentHash(replicas int) *ConsistentHash {
    return &ConsistentHash{
        nodes:    make([]Node, 0),
        replicas: replicas,
    }
}

func (ch *ConsistentHash) AddNode(nodeID string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    for i := 0; i < ch.replicas; i++ {
        replicaID := nodeID + ":" + strconv.Itoa(i)
        hash := ch.hash(replicaID)
        ch.nodes = append(ch.nodes, Node{ID: nodeID, Hash: hash})
    }
    
    sort.Slice(ch.nodes, func(i, j int) bool {
        return ch.nodes[i].Hash < ch.nodes[j].Hash
    })
}

func (ch *ConsistentHash) RemoveNode(nodeID string) {
    ch.mutex.Lock()
    defer ch.mutex.Unlock()
    
    var newNodes []Node
    for _, node := range ch.nodes {
        if node.ID != nodeID {
            newNodes = append(newNodes, node)
        }
    }
    ch.nodes = newNodes
}

func (ch *ConsistentHash) GetNode(key string) string {
    ch.mutex.RLock()
    defer ch.mutex.RUnlock()
    
    if len(ch.nodes) == 0 {
        return ""
    }
    
    hash := ch.hash(key)
    
    for _, node := range ch.nodes {
        if node.Hash >= hash {
            return node.ID
        }
    }
    
    return ch.nodes[0].ID
}

func (ch *ConsistentHash) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

type DistributedCache struct {
    hash     *ConsistentHash
    caches   map[string]map[string]string
    mutex    sync.RWMutex
}

func NewDistributedCache(replicas int) *DistributedCache {
    return &DistributedCache{
        hash:   NewConsistentHash(replicas),
        caches: make(map[string]map[string]string),
    }
}

func (dc *DistributedCache) AddNode(nodeID string) {
    dc.hash.AddNode(nodeID)
    dc.mutex.Lock()
    dc.caches[nodeID] = make(map[string]string)
    dc.mutex.Unlock()
}

func (dc *DistributedCache) Set(key, value string) {
    nodeID := dc.hash.GetNode(key)
    dc.mutex.Lock()
    if cache, exists := dc.caches[nodeID]; exists {
        cache[key] = value
    }
    dc.mutex.Unlock()
}

func (dc *DistributedCache) Get(key string) (string, bool) {
    nodeID := dc.hash.GetNode(key)
    dc.mutex.RLock()
    defer dc.mutex.RUnlock()
    
    if cache, exists := dc.caches[nodeID]; exists {
        value, found := cache[key]
        return value, found
    }
    
    return "", false
}

func main() {
    cache := NewDistributedCache(3)
    
    // Add nodes
    cache.AddNode("node1")
    cache.AddNode("node2")
    cache.AddNode("node3")
    
    // Set values
    cache.Set("key1", "value1")
    cache.Set("key2", "value2")
    cache.Set("key3", "value3")
    
    // Get values
    for _, key := range []string{"key1", "key2", "key3"} {
        if value, found := cache.Get(key); found {
            fmt.Printf("Key: %s, Value: %s\n", key, value)
        } else {
            fmt.Printf("Key: %s not found\n", key)
        }
    }
}
```

---

## 📚 Additional Resources

### **Books**
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms/) - Cormen, Leiserson, Rivest, Stein
- [Algorithm Design Manual](https://www.algorist.com/) - Steven Skiena
- [Competitive Programming](https://cpbook.net/) - Steven Halim

### **Online Resources**
- [Google's Technical Interview Guide](https://www.google.com/about/careers/students/guide-to-technical-development.html/)
- [LeetCode](https://leetcode.com/) - Practice problems
- [Codeforces](https://codeforces.com/) - Competitive programming

### **Video Resources**
- [MIT 6.006 Introduction to Algorithms](https://www.youtube.com/playlist?list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb/)
- [Stanford CS161 Design and Analysis of Algorithms](https://www.youtube.com/playlist?list=PLXFMmlk03Dt7Q0xr1PIAriY5623cKiH7V/)

---

*This comprehensive guide covers advanced algorithms and data structures essential for Google interviews, with practical Go implementations and real-world examples.*
