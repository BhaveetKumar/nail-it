---
# Auto-generated front matter
Title: Advanced Algorithms And Data Structures
LastUpdated: 2025-11-06T20:45:58.679117
Tags: []
Status: draft
---

# 🧮 **Advanced Algorithms and Data Structures**

## 📊 **Complete Guide to Advanced CS Concepts**

---

## 🎯 **1. Advanced Graph Algorithms**

### **Graph Algorithms Implementation**

```go
package main

import (
    "container/heap"
    "fmt"
    "math"
)

// Advanced Graph Algorithms
type Graph struct {
    vertices map[int]*Vertex
    edges    []*Edge
    directed bool
}

type Vertex struct {
    ID       int
    Data     interface{}
    Adjacent []*Edge
    Visited  bool
    Distance int
    Parent   *Vertex
}

type Edge struct {
    From     *Vertex
    To       *Vertex
    Weight   int
    Capacity int
    Flow     int
}

// Dijkstra's Algorithm
func (g *Graph) Dijkstra(startID int) map[int]int {
    distances := make(map[int]int)
    pq := make(PriorityQueue, 0)
    heap.Init(&pq)

    // Initialize distances
    for id := range g.vertices {
        distances[id] = math.MaxInt32
    }
    distances[startID] = 0

    // Add start vertex to priority queue
    heap.Push(&pq, &Item{
        vertex:   g.vertices[startID],
        priority: 0,
    })

    for pq.Len() > 0 {
        item := heap.Pop(&pq).(*Item)
        current := item.vertex

        if current.Visited {
            continue
        }

        current.Visited = true

        // Check all adjacent vertices
        for _, edge := range current.Adjacent {
            neighbor := edge.To
            if !neighbor.Visited {
                newDist := distances[current.ID] + edge.Weight
                if newDist < distances[neighbor.ID] {
                    distances[neighbor.ID] = newDist
                    neighbor.Parent = current
                    heap.Push(&pq, &Item{
                        vertex:   neighbor,
                        priority: newDist,
                    })
                }
            }
        }
    }

    return distances
}

// Bellman-Ford Algorithm
func (g *Graph) BellmanFord(startID int) (map[int]int, bool) {
    distances := make(map[int]int)

    // Initialize distances
    for id := range g.vertices {
        distances[id] = math.MaxInt32
    }
    distances[startID] = 0

    // Relax edges V-1 times
    for i := 0; i < len(g.vertices)-1; i++ {
        for _, edge := range g.edges {
            if distances[edge.From.ID] != math.MaxInt32 &&
                distances[edge.From.ID]+edge.Weight < distances[edge.To.ID] {
                distances[edge.To.ID] = distances[edge.From.ID] + edge.Weight
                edge.To.Parent = edge.From
            }
        }
    }

    // Check for negative cycles
    for _, edge := range g.edges {
        if distances[edge.From.ID] != math.MaxInt32 &&
            distances[edge.From.ID]+edge.Weight < distances[edge.To.ID] {
            return nil, false // Negative cycle detected
        }
    }

    return distances, true
}

// Floyd-Warshall Algorithm
func (g *Graph) FloydWarshall() [][]int {
    n := len(g.vertices)
    dist := make([][]int, n)

    // Initialize distance matrix
    for i := 0; i < n; i++ {
        dist[i] = make([]int, n)
        for j := 0; j < n; j++ {
            if i == j {
                dist[i][j] = 0
            } else {
                dist[i][j] = math.MaxInt32
            }
        }
    }

    // Set initial distances
    for _, edge := range g.edges {
        dist[edge.From.ID][edge.To.ID] = edge.Weight
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

// A* Algorithm
func (g *Graph) AStar(startID, goalID int, heuristic func(int, int) int) []int {
    openSet := make(PriorityQueue, 0)
    heap.Init(&openSet)

    gScore := make(map[int]int)
    fScore := make(map[int]int)
    cameFrom := make(map[int]*Vertex)

    // Initialize scores
    for id := range g.vertices {
        gScore[id] = math.MaxInt32
        fScore[id] = math.MaxInt32
    }

    gScore[startID] = 0
    fScore[startID] = heuristic(startID, goalID)

    heap.Push(&openSet, &Item{
        vertex:   g.vertices[startID],
        priority: fScore[startID],
    })

    for openSet.Len() > 0 {
        current := heap.Pop(&openSet).(*Item).vertex

        if current.ID == goalID {
            return g.reconstructPath(cameFrom, current.ID)
        }

        for _, edge := range current.Adjacent {
            neighbor := edge.To
            tentativeGScore := gScore[current.ID] + edge.Weight

            if tentativeGScore < gScore[neighbor.ID] {
                cameFrom[neighbor.ID] = current
                gScore[neighbor.ID] = tentativeGScore
                fScore[neighbor.ID] = gScore[neighbor.ID] + heuristic(neighbor.ID, goalID)

                heap.Push(&openSet, &Item{
                    vertex:   neighbor,
                    priority: fScore[neighbor.ID],
                })
            }
        }
    }

    return nil // No path found
}

func (g *Graph) reconstructPath(cameFrom map[int]*Vertex, currentID int) []int {
    path := []int{currentID}

    for {
        if parent, exists := cameFrom[currentID]; exists {
            path = append([]int{parent.ID}, path...)
            currentID = parent.ID
        } else {
            break
        }
    }

    return path
}

// Priority Queue for Dijkstra and A*
type PriorityQueue []*Item

type Item struct {
    vertex   *Vertex
    priority int
    index    int
}

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].priority < pq[j].priority
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

// Example usage
func main() {
    // Create graph
    graph := &Graph{
        vertices: make(map[int]*Vertex),
        edges:    make([]*Edge, 0),
        directed: true,
    }

    // Add vertices
    for i := 0; i < 5; i++ {
        graph.vertices[i] = &Vertex{
            ID:       i,
            Adjacent: make([]*Edge, 0),
        }
    }

    // Add edges
    edges := []struct {
        from, to, weight int
    }{
        {0, 1, 4},
        {0, 2, 2},
        {1, 2, 1},
        {1, 3, 5},
        {2, 3, 8},
        {2, 4, 10},
        {3, 4, 2},
    }

    for _, e := range edges {
        edge := &Edge{
            From:   graph.vertices[e.from],
            To:     graph.vertices[e.to],
            Weight: e.weight,
        }
        graph.edges = append(graph.edges, edge)
        graph.vertices[e.from].Adjacent = append(graph.vertices[e.from].Adjacent, edge)
    }

    // Run Dijkstra's algorithm
    distances := graph.Dijkstra(0)
    fmt.Printf("Dijkstra distances: %v\n", distances)

    // Run Bellman-Ford algorithm
    bfDistances, hasNegativeCycle := graph.BellmanFord(0)
    if hasNegativeCycle {
        fmt.Printf("Bellman-Ford distances: %v\n", bfDistances)
    } else {
        fmt.Println("Negative cycle detected")
    }

    // Run Floyd-Warshall algorithm
    fwDistances := graph.FloydWarshall()
    fmt.Printf("Floyd-Warshall distances: %v\n", fwDistances)
}
```

---

## 🎯 **2. Advanced Data Structures**

### **Trie, Segment Tree, and Fenwick Tree**

```go
package main

import (
    "fmt"
    "math"
)

// Trie (Prefix Tree)
type Trie struct {
    root *TrieNode
}

type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
    data     interface{}
}

func NewTrie() *Trie {
    return &Trie{
        root: &TrieNode{
            children: make(map[rune]*TrieNode),
        },
    }
}

func (t *Trie) Insert(word string, data interface{}) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{
                children: make(map[rune]*TrieNode),
            }
        }
        node = node.children[char]
    }
    node.isEnd = true
    node.data = data
}

func (t *Trie) Search(word string) (interface{}, bool) {
    node := t.root
    for _, char := range word {
        if node.children[char] == nil {
            return nil, false
        }
        node = node.children[char]
    }
    return node.data, node.isEnd
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

func (t *Trie) GetAllWordsWithPrefix(prefix string) []string {
    node := t.root
    for _, char := range prefix {
        if node.children[char] == nil {
            return nil
        }
        node = node.children[char]
    }

    var words []string
    t.dfs(node, prefix, &words)
    return words
}

func (t *Trie) dfs(node *TrieNode, current string, words *[]string) {
    if node.isEnd {
        *words = append(*words, current)
    }

    for char, child := range node.children {
        t.dfs(child, current+string(char), words)
    }
}

// Segment Tree
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

    st := &SegmentTree{
        tree: make([]int, 2*size),
        size: size,
        data: data,
    }

    st.build(0, 0, size-1)
    return st
}

func (st *SegmentTree) build(node, start, end int) {
    if start == end {
        if start < len(st.data) {
            st.tree[node] = st.data[start]
        }
        return
    }

    mid := (start + end) / 2
    st.build(2*node+1, start, mid)
    st.build(2*node+2, mid+1, end)
    st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
}

func (st *SegmentTree) Update(index, value int) {
    st.update(0, 0, st.size-1, index, value)
}

func (st *SegmentTree) update(node, start, end, index, value int) {
    if start == end {
        st.tree[node] = value
        return
    }

    mid := (start + end) / 2
    if index <= mid {
        st.update(2*node+1, start, mid, index, value)
    } else {
        st.update(2*node+2, mid+1, end, index, value)
    }
    st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
}

func (st *SegmentTree) Query(left, right int) int {
    return st.query(0, 0, st.size-1, left, right)
}

func (st *SegmentTree) query(node, start, end, left, right int) int {
    if right < start || left > end {
        return 0
    }

    if left <= start && end <= right {
        return st.tree[node]
    }

    mid := (start + end) / 2
    leftSum := st.query(2*node+1, start, mid, left, right)
    rightSum := st.query(2*node+2, mid+1, end, left, right)
    return leftSum + rightSum
}

// Fenwick Tree (Binary Indexed Tree)
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
    index++
    for index <= ft.size {
        ft.tree[index] += delta
        index += index & (-index)
    }
}

func (ft *FenwickTree) Query(index int) int {
    index++
    sum := 0
    for index > 0 {
        sum += ft.tree[index]
        index -= index & (-index)
    }
    return sum
}

func (ft *FenwickTree) RangeQuery(left, right int) int {
    return ft.Query(right) - ft.Query(left-1)
}

// Disjoint Set Union (Union-Find)
type DSU struct {
    parent []int
    rank   []int
    size   []int
}

func NewDSU(n int) *DSU {
    parent := make([]int, n)
    rank := make([]int, n)
    size := make([]int, n)

    for i := 0; i < n; i++ {
        parent[i] = i
        rank[i] = 0
        size[i] = 1
    }

    return &DSU{
        parent: parent,
        rank:   rank,
        size:   size,
    }
}

func (dsu *DSU) Find(x int) int {
    if dsu.parent[x] != x {
        dsu.parent[x] = dsu.Find(dsu.parent[x])
    }
    return dsu.parent[x]
}

func (dsu *DSU) Union(x, y int) {
    rootX := dsu.Find(x)
    rootY := dsu.Find(y)

    if rootX == rootY {
        return
    }

    if dsu.rank[rootX] < dsu.rank[rootY] {
        dsu.parent[rootX] = rootY
        dsu.size[rootY] += dsu.size[rootX]
    } else if dsu.rank[rootX] > dsu.rank[rootY] {
        dsu.parent[rootY] = rootX
        dsu.size[rootX] += dsu.size[rootY]
    } else {
        dsu.parent[rootY] = rootX
        dsu.rank[rootX]++
        dsu.size[rootX] += dsu.size[rootY]
    }
}

func (dsu *DSU) Connected(x, y int) bool {
    return dsu.Find(x) == dsu.Find(y)
}

func (dsu *DSU) GetSize(x int) int {
    return dsu.size[dsu.Find(x)]
}

// Example usage
func main() {
    // Test Trie
    trie := NewTrie()
    trie.Insert("hello", "world")
    trie.Insert("hi", "there")
    trie.Insert("help", "me")

    if data, found := trie.Search("hello"); found {
        fmt.Printf("Found: %v\n", data)
    }

    words := trie.GetAllWordsWithPrefix("he")
    fmt.Printf("Words with prefix 'he': %v\n", words)

    // Test Segment Tree
    data := []int{1, 3, 5, 7, 9, 11}
    st := NewSegmentTree(data)

    sum := st.Query(1, 3)
    fmt.Printf("Sum from index 1 to 3: %d\n", sum)

    st.Update(1, 10)
    sum = st.Query(1, 3)
    fmt.Printf("Sum after update: %d\n", sum)

    // Test Fenwick Tree
    ft := NewFenwickTree(6)
    for i, val := range data {
        ft.Update(i, val)
    }

    sum = ft.RangeQuery(1, 3)
    fmt.Printf("Fenwick Tree sum from 1 to 3: %d\n", sum)

    // Test DSU
    dsu := NewDSU(5)
    dsu.Union(0, 1)
    dsu.Union(2, 3)
    dsu.Union(1, 2)

    fmt.Printf("Are 0 and 3 connected? %v\n", dsu.Connected(0, 3))
    fmt.Printf("Size of component containing 0: %d\n", dsu.GetSize(0))
}
```

---

## 🎯 **3. Dynamic Programming Patterns**

### **Advanced DP Techniques**

```go
package main

import (
    "fmt"
    "math"
)

// Longest Common Subsequence
func LCS(s1, s2 string) int {
    m, n := len(s1), len(s2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }

    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }

    return dp[m][n]
}

// Longest Increasing Subsequence
func LIS(nums []int) int {
    n := len(nums)
    if n == 0 {
        return 0
    }

    dp := make([]int, n)
    dp[0] = 1
    maxLen := 1

    for i := 1; i < n; i++ {
        dp[i] = 1
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
        maxLen = max(maxLen, dp[i])
    }

    return maxLen
}

// Edit Distance (Levenshtein Distance)
func EditDistance(s1, s2 string) int {
    m, n := len(s1), len(s2)
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
            if s1[i-1] == s2[j-1] {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = min(
                    dp[i-1][j]+1,    // deletion
                    dp[i][j-1]+1,    // insertion
                    dp[i-1][j-1]+1,  // substitution
                )
            }
        }
    }

    return dp[m][n]
}

// Knapsack Problem
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

// Coin Change Problem
func CoinChange(coins []int, amount int) int {
    dp := make([]int, amount+1)
    for i := range dp {
        dp[i] = math.MaxInt32
    }
    dp[0] = 0

    for i := 1; i <= amount; i++ {
        for _, coin := range coins {
            if coin <= i {
                dp[i] = min(dp[i], dp[i-coin]+1)
            }
        }
    }

    if dp[amount] == math.MaxInt32 {
        return -1
    }
    return dp[amount]
}

// Longest Palindromic Subsequence
func LongestPalindromicSubsequence(s string) int {
    n := len(s)
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, n)
    }

    // Single characters are palindromes of length 1
    for i := 0; i < n; i++ {
        dp[i][i] = 1
    }

    // Check for palindromes of length 2
    for i := 0; i < n-1; i++ {
        if s[i] == s[i+1] {
            dp[i][i+1] = 2
        } else {
            dp[i][i+1] = 1
        }
    }

    // Check for palindromes of length 3 and more
    for length := 3; length <= n; length++ {
        for i := 0; i < n-length+1; i++ {
            j := i + length - 1
            if s[i] == s[j] {
                dp[i][j] = dp[i+1][j-1] + 2
            } else {
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
            }
        }
    }

    return dp[0][n-1]
}

// Matrix Chain Multiplication
func MatrixChainMultiplication(dims []int) int {
    n := len(dims) - 1
    dp := make([][]int, n)
    for i := range dp {
        dp[i] = make([]int, n)
    }

    for length := 2; length <= n; length++ {
        for i := 0; i < n-length+1; i++ {
            j := i + length - 1
            dp[i][j] = math.MaxInt32

            for k := i; k < j; k++ {
                cost := dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                if cost < dp[i][j] {
                    dp[i][j] = cost
                }
            }
        }
    }

    return dp[0][n-1]
}

// Helper functions
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

// Example usage
func main() {
    // Test LCS
    s1, s2 := "ABCDGH", "AEDFHR"
    lcs := LCS(s1, s2)
    fmt.Printf("LCS of '%s' and '%s': %d\n", s1, s2, lcs)

    // Test LIS
    nums := []int{10, 9, 2, 5, 3, 7, 101, 18}
    lis := LIS(nums)
    fmt.Printf("LIS of %v: %d\n", nums, lis)

    // Test Edit Distance
    s1, s2 = "kitten", "sitting"
    editDist := EditDistance(s1, s2)
    fmt.Printf("Edit distance between '%s' and '%s': %d\n", s1, s2, editDist)

    // Test Knapsack
    weights := []int{1, 3, 4, 5}
    values := []int{1, 4, 5, 7}
    capacity := 7
    knapsack := Knapsack(weights, values, capacity)
    fmt.Printf("Knapsack value: %d\n", knapsack)

    // Test Coin Change
    coins := []int{1, 3, 4}
    amount := 6
    coinChange := CoinChange(coins, amount)
    fmt.Printf("Minimum coins needed: %d\n", coinChange)

    // Test Longest Palindromic Subsequence
    s := "bbbab"
    lps := LongestPalindromicSubsequence(s)
    fmt.Printf("Longest palindromic subsequence of '%s': %d\n", s, lps)

    // Test Matrix Chain Multiplication
    dims := []int{1, 2, 3, 4, 5}
    mcm := MatrixChainMultiplication(dims)
    fmt.Printf("Minimum multiplications: %d\n", mcm)
}
```

---

## 🎯 **Key Takeaways from Advanced Algorithms and Data Structures**

### **1. Advanced Graph Algorithms**

- **Shortest Path**: Dijkstra, Bellman-Ford, Floyd-Warshall
- **Pathfinding**: A\* algorithm with heuristics
- **Priority Queues**: Efficient implementation for graph algorithms
- **Negative Cycles**: Detection and handling

### **2. Advanced Data Structures**

- **Trie**: Prefix tree for string operations
- **Segment Tree**: Range queries and updates
- **Fenwick Tree**: Efficient prefix sums
- **Union-Find**: Disjoint set operations

### **3. Dynamic Programming**

- **String Algorithms**: LCS, Edit Distance, LPS
- **Optimization**: Knapsack, Coin Change
- **Matrix Operations**: Chain multiplication
- **Pattern Recognition**: Common DP patterns

### **4. Production Considerations**

- **Time Complexity**: Understanding Big O notation
- **Space Complexity**: Memory optimization
- **Edge Cases**: Handling boundary conditions
- **Testing**: Comprehensive test coverage

---

**🎉 This comprehensive guide provides advanced algorithms and data structures with production-ready Go implementations for competitive programming and system design! 🚀**
