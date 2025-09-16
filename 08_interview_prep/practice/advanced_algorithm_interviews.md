# Advanced Algorithm Interviews

## Table of Contents
- [Introduction](#introduction)
- [Complex Data Structures](#complex-data-structures)
- [Advanced Graph Problems](#advanced-graph-problems)
- [Dynamic Programming](#dynamic-programming)
- [Greedy Algorithms](#greedy-algorithms)
- [Backtracking](#backtracking)
- [Mathematical Algorithms](#mathematical-algorithms)

## Introduction

Advanced algorithm interviews test your ability to solve complex problems using sophisticated algorithmic techniques and data structures.

## Complex Data Structures

### Design a LRU Cache

**Problem**: Design and implement a Least Recently Used (LRU) cache.

```go
// LRU Cache Implementation
type LRUCache struct {
    capacity int
    cache    map[int]*Node
    head     *Node
    tail     *Node
    mu       sync.RWMutex
}

type Node struct {
    key   int
    value int
    prev  *Node
    next  *Node
}

func NewLRUCache(capacity int) *LRUCache {
    cache := &LRUCache{
        capacity: capacity,
        cache:    make(map[int]*Node),
    }
    
    // Initialize dummy head and tail
    cache.head = &Node{}
    cache.tail = &Node{}
    cache.head.next = cache.tail
    cache.tail.prev = cache.head
    
    return cache
}

func (lru *LRUCache) Get(key int) int {
    lru.mu.Lock()
    defer lru.mu.Unlock()
    
    if node, exists := lru.cache[key]; exists {
        // Move to head (most recently used)
        lru.moveToHead(node)
        return node.value
    }
    
    return -1
}

func (lru *LRUCache) Put(key, value int) {
    lru.mu.Lock()
    defer lru.mu.Unlock()
    
    if node, exists := lru.cache[key]; exists {
        // Update existing node
        node.value = value
        lru.moveToHead(node)
    } else {
        // Create new node
        newNode := &Node{
            key:   key,
            value: value,
        }
        
        if len(lru.cache) >= lru.capacity {
            // Remove least recently used
            lru.removeTail()
        }
        
        lru.cache[key] = newNode
        lru.addToHead(newNode)
    }
}

func (lru *LRUCache) addToHead(node *Node) {
    node.prev = lru.head
    node.next = lru.head.next
    lru.head.next.prev = node
    lru.head.next = node
}

func (lru *LRUCache) removeNode(node *Node) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (lru *LRUCache) moveToHead(node *Node) {
    lru.removeNode(node)
    lru.addToHead(node)
}

func (lru *LRUCache) removeTail() {
    lastNode := lru.tail.prev
    lru.removeNode(lastNode)
    delete(lru.cache, lastNode.key)
}
```

### Design a Trie with Wildcard Search

**Problem**: Implement a trie that supports wildcard pattern matching.

```go
// Trie with Wildcard Search
type TrieNode struct {
    children map[rune]*TrieNode
    isEnd    bool
    word     string
}

type WildcardTrie struct {
    root *TrieNode
}

func NewWildcardTrie() *WildcardTrie {
    return &WildcardTrie{
        root: &TrieNode{
            children: make(map[rune]*TrieNode),
        },
    }
}

func (wt *WildcardTrie) Insert(word string) {
    node := wt.root
    for _, char := range word {
        if node.children[char] == nil {
            node.children[char] = &TrieNode{
                children: make(map[rune]*TrieNode),
            }
        }
        node = node.children[char]
    }
    node.isEnd = true
    node.word = word
}

func (wt *WildcardTrie) Search(word string) bool {
    return wt.searchHelper(wt.root, word, 0)
}

func (wt *WildcardTrie) searchHelper(node *TrieNode, word string, index int) bool {
    if index == len(word) {
        return node.isEnd
    }
    
    char := rune(word[index])
    
    if char == '.' {
        // Wildcard - try all children
        for _, child := range node.children {
            if wt.searchHelper(child, word, index+1) {
                return true
            }
        }
        return false
    } else {
        // Regular character
        if child, exists := node.children[char]; exists {
            return wt.searchHelper(child, word, index+1)
        }
        return false
    }
}

func (wt *WildcardTrie) FindAllMatches(pattern string) []string {
    var result []string
    wt.findAllMatchesHelper(wt.root, pattern, 0, &result)
    return result
}

func (wt *WildcardTrie) findAllMatchesHelper(node *TrieNode, pattern string, index int, result *[]string) {
    if index == len(pattern) {
        if node.isEnd {
            *result = append(*result, node.word)
        }
        return
    }
    
    char := rune(pattern[index])
    
    if char == '.' {
        // Wildcard - try all children
        for _, child := range node.children {
            wt.findAllMatchesHelper(child, pattern, index+1, result)
        }
    } else {
        // Regular character
        if child, exists := node.children[char]; exists {
            wt.findAllMatchesHelper(child, pattern, index+1, result)
        }
    }
}
```

## Advanced Graph Problems

### Critical Connections in a Network

**Problem**: Find all critical connections in a network (bridges).

```go
// Critical Connections (Bridges)
func findCriticalConnections(n int, connections [][]int) [][]int {
    // Build adjacency list
    graph := make([][]int, n)
    for _, conn := range connections {
        graph[conn[0]] = append(graph[conn[0]], conn[1])
        graph[conn[1]] = append(graph[conn[1]], conn[0])
    }
    
    var result [][]int
    discovery := make([]int, n)
    low := make([]int, n)
    time := 0
    
    var dfs func(int, int)
    dfs = func(u, parent int) {
        discovery[u] = time
        low[u] = time
        time++
        
        for _, v := range graph[u] {
            if v == parent {
                continue
            }
            
            if discovery[v] == -1 {
                // Not visited
                dfs(v, u)
                low[u] = min(low[u], low[v])
                
                // Check if edge u-v is a bridge
                if low[v] > discovery[u] {
                    result = append(result, []int{u, v})
                }
            } else {
                // Back edge
                low[u] = min(low[u], discovery[v])
            }
        }
    }
    
    // Initialize discovery times
    for i := 0; i < n; i++ {
        discovery[i] = -1
    }
    
    // DFS from each unvisited node
    for i := 0; i < n; i++ {
        if discovery[i] == -1 {
            dfs(i, -1)
        }
    }
    
    return result
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Alien Dictionary

**Problem**: Given a sorted dictionary of alien language, find the order of characters.

```go
// Alien Dictionary
func alienOrder(words []string) string {
    // Build graph
    graph := make(map[byte][]byte)
    inDegree := make(map[byte]int)
    
    // Initialize in-degree for all characters
    for _, word := range words {
        for _, char := range word {
            inDegree[byte(char)] = 0
        }
    }
    
    // Build edges
    for i := 0; i < len(words)-1; i++ {
        word1, word2 := words[i], words[i+1]
        
        // Check for invalid case: word1 is prefix of word2
        if len(word1) > len(word2) && word1[:len(word2)] == word2 {
            return ""
        }
        
        // Find first different character
        for j := 0; j < min(len(word1), len(word2)); j++ {
            if word1[j] != word2[j] {
                graph[word1[j]] = append(graph[word1[j]], word2[j])
                inDegree[word2[j]]++
                break
            }
        }
    }
    
    // Topological sort using Kahn's algorithm
    queue := make([]byte, 0)
    for char, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, char)
        }
    }
    
    var result []byte
    for len(queue) > 0 {
        char := queue[0]
        queue = queue[1:]
        result = append(result, char)
        
        for _, neighbor := range graph[char] {
            inDegree[neighbor]--
            if inDegree[neighbor] == 0 {
                queue = append(queue, neighbor)
            }
        }
    }
    
    // Check if all characters are included
    if len(result) != len(inDegree) {
        return ""
    }
    
    return string(result)
}
```

## Dynamic Programming

### Longest Increasing Path in Matrix

**Problem**: Find the longest increasing path in a matrix.

```go
// Longest Increasing Path in Matrix
func longestIncreasingPath(matrix [][]int) int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return 0
    }
    
    m, n := len(matrix), len(matrix[0])
    memo := make([][]int, m)
    for i := range memo {
        memo[i] = make([]int, n)
    }
    
    maxPath := 0
    directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
    
    var dfs func(int, int) int
    dfs = func(i, j int) int {
        if memo[i][j] != 0 {
            return memo[i][j]
        }
        
        maxLen := 1
        for _, dir := range directions {
            ni, nj := i+dir[0], j+dir[1]
            if ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni][nj] > matrix[i][j] {
                maxLen = max(maxLen, 1+dfs(ni, nj))
            }
        }
        
        memo[i][j] = maxLen
        return maxLen
    }
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            maxPath = max(maxPath, dfs(i, j))
        }
    }
    
    return maxPath
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Word Break II

**Problem**: Given a string and a dictionary, return all possible sentences.

```go
// Word Break II
func wordBreak(s string, wordDict []string) []string {
    wordSet := make(map[string]bool)
    for _, word := range wordDict {
        wordSet[word] = true
    }
    
    memo := make(map[string][]string)
    return wordBreakHelper(s, wordSet, memo)
}

func wordBreakHelper(s string, wordSet map[string]bool, memo map[string][]string) []string {
    if result, exists := memo[s]; exists {
        return result
    }
    
    var result []string
    
    if len(s) == 0 {
        result = append(result, "")
        return result
    }
    
    for i := 1; i <= len(s); i++ {
        prefix := s[:i]
        if wordSet[prefix] {
            suffixResults := wordBreakHelper(s[i:], wordSet, memo)
            for _, suffix := range suffixResults {
                if suffix == "" {
                    result = append(result, prefix)
                } else {
                    result = append(result, prefix+" "+suffix)
                }
            }
        }
    }
    
    memo[s] = result
    return result
}
```

## Greedy Algorithms

### Minimum Number of Arrows to Burst Balloons

**Problem**: Find minimum arrows to burst all balloons.

```go
// Minimum Arrows to Burst Balloons
func findMinArrowShots(points [][]int) int {
    if len(points) == 0 {
        return 0
    }
    
    // Sort by end points
    sort.Slice(points, func(i, j int) bool {
        return points[i][1] < points[j][1]
    })
    
    arrows := 1
    end := points[0][1]
    
    for i := 1; i < len(points); i++ {
        if points[i][0] > end {
            arrows++
            end = points[i][1]
        }
    }
    
    return arrows
}
```

### Reorganize String

**Problem**: Reorganize string so no two adjacent characters are the same.

```go
// Reorganize String
func reorganizeString(s string) string {
    // Count character frequencies
    count := make(map[byte]int)
    for i := 0; i < len(s); i++ {
        count[s[i]]++
    }
    
    // Use max heap
    heap := &MaxHeap{}
    heap.Init()
    
    for char, freq := range count {
        heap.Push(&CharFreq{char: char, freq: freq})
    }
    
    var result []byte
    var prev *CharFreq
    
    for heap.Len() > 0 {
        curr := heap.Pop().(*CharFreq)
        result = append(result, curr.char)
        curr.freq--
        
        if prev != nil && prev.freq > 0 {
            heap.Push(prev)
        }
        
        prev = curr
    }
    
    if len(result) != len(s) {
        return ""
    }
    
    return string(result)
}

type CharFreq struct {
    char byte
    freq int
}

type MaxHeap []*CharFreq

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].freq > h[j].freq }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
    *h = append(*h, x.(*CharFreq))
}

func (h *MaxHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func (h *MaxHeap) Init() {
    heap.Init(h)
}
```

## Backtracking

### N-Queens Problem

**Problem**: Place N queens on an N×N chessboard so no two queens attack each other.

```go
// N-Queens Problem
func solveNQueens(n int) [][]string {
    var result [][]string
    board := make([][]bool, n)
    for i := range board {
        board[i] = make([]bool, n)
    }
    
    var backtrack func(int)
    backtrack = func(row int) {
        if row == n {
            // Convert board to string representation
            solution := make([]string, n)
            for i := 0; i < n; i++ {
                var rowStr strings.Builder
                for j := 0; j < n; j++ {
                    if board[i][j] {
                        rowStr.WriteString("Q")
                    } else {
                        rowStr.WriteString(".")
                    }
                }
                solution[i] = rowStr.String()
            }
            result = append(result, solution)
            return
        }
        
        for col := 0; col < n; col++ {
            if isValid(board, row, col) {
                board[row][col] = true
                backtrack(row + 1)
                board[row][col] = false
            }
        }
    }
    
    backtrack(0)
    return result
}

func isValid(board [][]bool, row, col int) bool {
    n := len(board)
    
    // Check column
    for i := 0; i < row; i++ {
        if board[i][col] {
            return false
        }
    }
    
    // Check diagonal (top-left to bottom-right)
    for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
        if board[i][j] {
            return false
        }
    }
    
    // Check diagonal (top-right to bottom-left)
    for i, j := row-1, col+1; i >= 0 && j < n; i, j = i-1, j+1 {
        if board[i][j] {
            return false
        }
    }
    
    return true
}
```

### Word Search II

**Problem**: Find all words from a dictionary that exist in a 2D board.

```go
// Word Search II
func findWords(board [][]byte, words []string) []string {
    // Build trie
    trie := &TrieNode{
        children: make(map[byte]*TrieNode),
    }
    
    for _, word := range words {
        node := trie
        for _, char := range word {
            if node.children[char] == nil {
                node.children[char] = &TrieNode{
                    children: make(map[byte]*TrieNode),
                }
            }
            node = node.children[char]
        }
        node.word = word
    }
    
    var result []string
    m, n := len(board), len(board[0])
    visited := make([][]bool, m)
    for i := range visited {
        visited[i] = make([]bool, n)
    }
    
    directions := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
    
    var dfs func(int, int, *TrieNode)
    dfs = func(i, j int, node *TrieNode) {
        if i < 0 || i >= m || j < 0 || j >= n || visited[i][j] {
            return
        }
        
        char := board[i][j]
        if node.children[char] == nil {
            return
        }
        
        node = node.children[char]
        if node.word != "" {
            result = append(result, node.word)
            node.word = "" // Avoid duplicates
        }
        
        visited[i][j] = true
        for _, dir := range directions {
            dfs(i+dir[0], j+dir[1], node)
        }
        visited[i][j] = false
    }
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            dfs(i, j, trie)
        }
    }
    
    return result
}
```

## Mathematical Algorithms

### Count Primes

**Problem**: Count the number of prime numbers less than n.

```go
// Count Primes using Sieve of Eratosthenes
func countPrimes(n int) int {
    if n <= 2 {
        return 0
    }
    
    isPrime := make([]bool, n)
    for i := 2; i < n; i++ {
        isPrime[i] = true
    }
    
    for i := 2; i*i < n; i++ {
        if isPrime[i] {
            for j := i * i; j < n; j += i {
                isPrime[j] = false
            }
        }
    }
    
    count := 0
    for i := 2; i < n; i++ {
        if isPrime[i] {
            count++
        }
    }
    
    return count
}
```

### Pow(x, n)

**Problem**: Implement pow(x, n) efficiently.

```go
// Fast Power Algorithm
func myPow(x float64, n int) float64 {
    if n == 0 {
        return 1
    }
    
    if n < 0 {
        x = 1 / x
        n = -n
    }
    
    result := 1.0
    for n > 0 {
        if n&1 == 1 {
            result *= x
        }
        x *= x
        n >>= 1
    }
    
    return result
}
```

## Conclusion

Advanced algorithm interviews test:

1. **Complexity Analysis**: Understanding time and space complexity
2. **Data Structures**: Mastery of advanced data structures
3. **Algorithm Design**: Ability to design efficient algorithms
4. **Problem Solving**: Breaking down complex problems
5. **Optimization**: Finding optimal solutions
6. **Implementation**: Writing clean, efficient code
7. **Edge Cases**: Handling all possible scenarios

Preparing for these advanced scenarios demonstrates your readiness for senior engineering roles and complex algorithmic challenges.

## Additional Resources

- [Advanced Algorithms](https://www.advancedalgorithms.com/)
- [Complex Data Structures](https://www.complexdatastructures.com/)
- [Graph Algorithms](https://www.graphalgorithms.com/)
- [Dynamic Programming](https://www.dynamicprogramming.com/)
- [Greedy Algorithms](https://www.greedyalgorithms.com/)
- [Backtracking](https://www.backtracking.com/)
- [Mathematical Algorithms](https://www.mathematicalalgorithms.com/)
- [Algorithm Design](https://www.algorithmdesign.com/)
