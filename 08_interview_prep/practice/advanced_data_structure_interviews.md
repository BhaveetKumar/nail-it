# Advanced Data Structure Interviews

## Table of Contents
- [Introduction](#introduction)
- [Advanced Trees](#advanced-trees)
- [Advanced Graphs](#advanced-graphs)
- [Advanced Hash Tables](#advanced-hash-tables)
- [Advanced Heaps](#advanced-heaps)
- [Advanced Tries](#advanced-tries)
- [Advanced Segment Trees](#advanced-segment-trees)
- [Advanced Fenwick Trees](#advanced-fenwick-trees)
- [Advanced Disjoint Sets](#advanced-disjoint-sets)

## Introduction

Advanced data structure interviews test your understanding of complex data structures and their applications in solving real-world problems.

## Advanced Trees

### Red-Black Tree Implementation

**Problem**: Implement a Red-Black Tree with insertion and deletion.

```go
// Red-Black Tree Implementation
type Color bool

const (
    RED   Color = true
    BLACK Color = false
)

type RBNode struct {
    key    int
    value  interface{}
    color  Color
    left   *RBNode
    right  *RBNode
    parent *RBNode
}

type RedBlackTree struct {
    root *RBNode
    nil  *RBNode
}

func NewRedBlackTree() *RedBlackTree {
    nilNode := &RBNode{color: BLACK}
    return &RedBlackTree{
        root: nilNode,
        nil:  nilNode,
    }
}

func (rbt *RedBlackTree) Insert(key int, value interface{}) {
    newNode := &RBNode{
        key:    key,
        value:  value,
        color:  RED,
        left:   rbt.nil,
        right:  rbt.nil,
        parent: rbt.nil,
    }
    
    // Standard BST insertion
    var y *RBNode = rbt.nil
    x := rbt.root
    
    for x != rbt.nil {
        y = x
        if newNode.key < x.key {
            x = x.left
        } else {
            x = x.right
        }
    }
    
    newNode.parent = y
    if y == rbt.nil {
        rbt.root = newNode
    } else if newNode.key < y.key {
        y.left = newNode
    } else {
        y.right = newNode
    }
    
    // Fix Red-Black properties
    rbt.insertFixup(newNode)
}

func (rbt *RedBlackTree) insertFixup(z *RBNode) {
    for z.parent.color == RED {
        if z.parent == z.parent.parent.left {
            y := z.parent.parent.right
            if y.color == RED {
                // Case 1: Uncle is red
                z.parent.color = BLACK
                y.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            } else {
                if z == z.parent.right {
                    // Case 2: Uncle is black, z is right child
                    z = z.parent
                    rbt.leftRotate(z)
                }
                // Case 3: Uncle is black, z is left child
                z.parent.color = BLACK
                z.parent.parent.color = RED
                rbt.rightRotate(z.parent.parent)
            }
        } else {
            // Symmetric case
            y := z.parent.parent.left
            if y.color == RED {
                z.parent.color = BLACK
                y.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            } else {
                if z == z.parent.left {
                    z = z.parent
                    rbt.rightRotate(z)
                }
                z.parent.color = BLACK
                z.parent.parent.color = RED
                rbt.leftRotate(z.parent.parent)
            }
        }
    }
    rbt.root.color = BLACK
}

func (rbt *RedBlackTree) leftRotate(x *RBNode) {
    y := x.right
    x.right = y.left
    
    if y.left != rbt.nil {
        y.left.parent = x
    }
    
    y.parent = x.parent
    
    if x.parent == rbt.nil {
        rbt.root = y
    } else if x == x.parent.left {
        x.parent.left = y
    } else {
        x.parent.right = y
    }
    
    y.left = x
    x.parent = y
}

func (rbt *RedBlackTree) rightRotate(y *RBNode) {
    x := y.left
    y.left = x.right
    
    if x.right != rbt.nil {
        x.right.parent = y
    }
    
    x.parent = y.parent
    
    if y.parent == rbt.nil {
        rbt.root = x
    } else if y == y.parent.left {
        y.parent.left = x
    } else {
        y.parent.right = x
    }
    
    x.right = y
    y.parent = x
}

func (rbt *RedBlackTree) Search(key int) interface{} {
    node := rbt.root
    for node != rbt.nil {
        if key == node.key {
            return node.value
        } else if key < node.key {
            node = node.left
        } else {
            node = node.right
        }
    }
    return nil
}
```

### B-Tree Implementation

**Problem**: Implement a B-Tree for database indexing.

```go
// B-Tree Implementation
type BTreeNode struct {
    keys     []int
    values   []interface{}
    children []*BTreeNode
    leaf     bool
    n        int // number of keys
}

type BTree struct {
    root *BTreeNode
    t    int // minimum degree
}

func NewBTree(t int) *BTree {
    return &BTree{
        root: nil,
        t:    t,
    }
}

func (bt *BTree) Search(key int) interface{} {
    if bt.root == nil {
        return nil
    }
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) interface{} {
    i := 0
    for i < node.n && key > node.keys[i] {
        i++
    }
    
    if i < node.n && key == node.keys[i] {
        return node.values[i]
    }
    
    if node.leaf {
        return nil
    }
    
    return bt.searchNode(node.children[i], key)
}

func (bt *BTree) Insert(key int, value interface{}) {
    if bt.root == nil {
        bt.root = &BTreeNode{
            keys:     make([]int, 2*bt.t-1),
            values:   make([]interface{}, 2*bt.t-1),
            children: make([]*BTreeNode, 2*bt.t),
            leaf:     true,
            n:        0,
        }
        bt.root.keys[0] = key
        bt.root.values[0] = value
        bt.root.n = 1
    } else {
        if bt.root.n == 2*bt.t-1 {
            // Root is full, need to split
            newRoot := &BTreeNode{
                keys:     make([]int, 2*bt.t-1),
                values:   make([]interface{}, 2*bt.t-1),
                children: make([]*BTreeNode, 2*bt.t),
                leaf:     false,
                n:        0,
            }
            newRoot.children[0] = bt.root
            bt.splitChild(newRoot, 0)
            bt.root = newRoot
        }
        bt.insertNonFull(bt.root, key, value)
    }
}

func (bt *BTree) insertNonFull(node *BTreeNode, key int, value interface{}) {
    i := node.n - 1
    
    if node.leaf {
        // Insert into leaf node
        for i >= 0 && key < node.keys[i] {
            node.keys[i+1] = node.keys[i]
            node.values[i+1] = node.values[i]
            i--
        }
        node.keys[i+1] = key
        node.values[i+1] = value
        node.n++
    } else {
        // Find child to insert into
        for i >= 0 && key < node.keys[i] {
            i--
        }
        i++
        
        if node.children[i].n == 2*bt.t-1 {
            // Child is full, split it
            bt.splitChild(node, i)
            if key > node.keys[i] {
                i++
            }
        }
        bt.insertNonFull(node.children[i], key, value)
    }
}

func (bt *BTree) splitChild(parent *BTreeNode, index int) {
    t := bt.t
    fullChild := parent.children[index]
    
    // Create new node for right half
    newChild := &BTreeNode{
        keys:     make([]int, 2*t-1),
        values:   make([]interface{}, 2*t-1),
        children: make([]*BTreeNode, 2*t),
        leaf:     fullChild.leaf,
        n:        t - 1,
    }
    
    // Copy right half of keys and values
    for j := 0; j < t-1; j++ {
        newChild.keys[j] = fullChild.keys[j+t]
        newChild.values[j] = fullChild.values[j+t]
    }
    
    // Copy right half of children if not leaf
    if !fullChild.leaf {
        for j := 0; j < t; j++ {
            newChild.children[j] = fullChild.children[j+t]
        }
    }
    
    // Update full child's key count
    fullChild.n = t - 1
    
    // Shift parent's children to make room
    for j := parent.n; j >= index+1; j-- {
        parent.children[j+1] = parent.children[j]
    }
    
    // Link new child to parent
    parent.children[index+1] = newChild
    
    // Shift parent's keys and values
    for j := parent.n - 1; j >= index; j-- {
        parent.keys[j+1] = parent.keys[j]
        parent.values[j+1] = parent.values[j]
    }
    
    // Move middle key to parent
    parent.keys[index] = fullChild.keys[t-1]
    parent.values[index] = fullChild.values[t-1]
    parent.n++
}
```

## Advanced Graphs

### Union-Find with Path Compression

**Problem**: Implement Union-Find with path compression and union by rank.

```go
// Union-Find with Path Compression
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
        rank[i] = 0
    }
    
    return &UnionFind{
        parent: parent,
        rank:   rank,
        count:  n,
    }
}

func (uf *UnionFind) Find(x int) int {
    if uf.parent[x] != x {
        // Path compression
        uf.parent[x] = uf.Find(uf.parent[x])
    }
    return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
    rootX := uf.Find(x)
    rootY := uf.Find(y)
    
    if rootX == rootY {
        return
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
}

func (uf *UnionFind) Connected(x, y int) bool {
    return uf.Find(x) == uf.Find(y)
}

func (uf *UnionFind) Count() int {
    return uf.count
}
```

### Topological Sort with Cycle Detection

**Problem**: Implement topological sort with cycle detection.

```go
// Topological Sort with Cycle Detection
func topologicalSort(n int, edges [][]int) ([]int, bool) {
    // Build adjacency list
    graph := make([][]int, n)
    inDegree := make([]int, n)
    
    for _, edge := range edges {
        from, to := edge[0], edge[1]
        graph[from] = append(graph[from], to)
        inDegree[to]++
    }
    
    // Kahn's algorithm
    queue := make([]int, 0)
    for i := 0; i < n; i++ {
        if inDegree[i] == 0 {
            queue = append(queue, i)
        }
    }
    
    var result []int
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        result = append(result, node)
        
        for _, neighbor := range graph[node] {
            inDegree[neighbor]--
            if inDegree[neighbor] == 0 {
                queue = append(queue, neighbor)
            }
        }
    }
    
    // Check for cycle
    if len(result) != n {
        return nil, false // Cycle detected
    }
    
    return result, true
}
```

## Advanced Hash Tables

### Consistent Hashing

**Problem**: Implement consistent hashing for distributed systems.

```go
// Consistent Hashing
import (
    "crypto/md5"
    "fmt"
    "sort"
    "strconv"
)

type HashRing struct {
    nodes   []string
    hashes  []uint32
    nodeMap map[uint32]string
}

func NewHashRing() *HashRing {
    return &HashRing{
        nodes:   make([]string, 0),
        hashes:  make([]uint32, 0),
        nodeMap: make(map[uint32]string),
    }
}

func (hr *HashRing) AddNode(node string) {
    hr.nodes = append(hr.nodes, node)
    
    // Add multiple virtual nodes for better distribution
    for i := 0; i < 150; i++ {
        virtualNode := fmt.Sprintf("%s:%d", node, i)
        hash := hr.hash(virtualNode)
        hr.hashes = append(hr.hashes, hash)
        hr.nodeMap[hash] = node
    }
    
    sort.Slice(hr.hashes, func(i, j int) bool {
        return hr.hashes[i] < hr.hashes[j]
    })
}

func (hr *HashRing) RemoveNode(node string) {
    // Remove all virtual nodes
    newHashes := make([]uint32, 0)
    for _, hash := range hr.hashes {
        if hr.nodeMap[hash] != node {
            newHashes = append(newHashes, hash)
        } else {
            delete(hr.nodeMap, hash)
        }
    }
    hr.hashes = newHashes
    
    // Remove from nodes list
    for i, n := range hr.nodes {
        if n == node {
            hr.nodes = append(hr.nodes[:i], hr.nodes[i+1:]...)
            break
        }
    }
}

func (hr *HashRing) GetNode(key string) string {
    if len(hr.hashes) == 0 {
        return ""
    }
    
    hash := hr.hash(key)
    
    // Find the first node with hash >= key hash
    idx := sort.Search(len(hr.hashes), func(i int) bool {
        return hr.hashes[i] >= hash
    })
    
    // Wrap around if necessary
    if idx == len(hr.hashes) {
        idx = 0
    }
    
    return hr.nodeMap[hr.hashes[idx]]
}

func (hr *HashRing) hash(key string) uint32 {
    h := md5.Sum([]byte(key))
    return uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}
```

### Bloom Filter

**Problem**: Implement a Bloom filter for probabilistic set membership.

```go
// Bloom Filter
import (
    "hash/fnv"
    "math"
)

type BloomFilter struct {
    bitArray []bool
    size     int
    hashFuncs []func(string) int
}

func NewBloomFilter(expectedItems int, falsePositiveRate float64) *BloomFilter {
    // Calculate optimal size and number of hash functions
    size := int(-float64(expectedItems) * math.Log(falsePositiveRate) / (math.Log(2) * math.Log(2)))
    numHashFuncs := int(float64(size) / float64(expectedItems) * math.Log(2))
    
    // Create hash functions
    hashFuncs := make([]func(string) int, numHashFuncs)
    for i := 0; i < numHashFuncs; i++ {
        seed := uint32(i)
        hashFuncs[i] = func(s string) int {
            h := fnv.New32a()
            h.Write([]byte(s))
            h.Write([]byte{byte(seed)})
            return int(h.Sum32()) % size
        }
    }
    
    return &BloomFilter{
        bitArray:  make([]bool, size),
        size:      size,
        hashFuncs: hashFuncs,
    }
}

func (bf *BloomFilter) Add(item string) {
    for _, hashFunc := range bf.hashFuncs {
        index := hashFunc(item)
        bf.bitArray[index] = true
    }
}

func (bf *BloomFilter) Contains(item string) bool {
    for _, hashFunc := range bf.hashFuncs {
        index := hashFunc(item)
        if !bf.bitArray[index] {
            return false
        }
    }
    return true
}

func (bf *BloomFilter) FalsePositiveRate() float64 {
    // Estimate false positive rate
    setBits := 0
    for _, bit := range bf.bitArray {
        if bit {
            setBits++
        }
    }
    
    ratio := float64(setBits) / float64(bf.size)
    return math.Pow(ratio, float64(len(bf.hashFuncs)))
}
```

## Advanced Heaps

### Binomial Heap

**Problem**: Implement a binomial heap for priority queues.

```go
// Binomial Heap
type BinomialNode struct {
    key     int
    degree  int
    parent  *BinomialNode
    child   *BinomialNode
    sibling *BinomialNode
}

type BinomialHeap struct {
    head *BinomialNode
}

func NewBinomialHeap() *BinomialHeap {
    return &BinomialHeap{head: nil}
}

func (bh *BinomialHeap) Insert(key int) {
    newNode := &BinomialNode{key: key, degree: 0}
    bh.union(newNode)
}

func (bh *BinomialHeap) ExtractMin() int {
    if bh.head == nil {
        return -1 // Empty heap
    }
    
    // Find minimum node
    minNode := bh.head
    prev := (*BinomialNode)(nil)
    current := bh.head
    
    for current.sibling != nil {
        if current.sibling.key < minNode.key {
            minNode = current.sibling
            prev = current
        }
        current = current.sibling
    }
    
    // Remove min node
    if prev == nil {
        bh.head = minNode.sibling
    } else {
        prev.sibling = minNode.sibling
    }
    
    // Reverse children and add to heap
    if minNode.child != nil {
        reversed := bh.reverseList(minNode.child)
        bh.union(reversed)
    }
    
    return minNode.key
}

func (bh *BinomialHeap) union(other *BinomialNode) {
    // Merge two binomial heaps
    merged := bh.merge(bh.head, other)
    bh.head = merged
    
    if bh.head == nil {
        return
    }
    
    // Consolidate trees of same degree
    prev := (*BinomialNode)(nil)
    current := bh.head
    next := current.sibling
    
    for next != nil {
        if current.degree != next.degree || 
           (next.sibling != nil && next.sibling.degree == current.degree) {
            prev = current
            current = next
        } else if current.key <= next.key {
            current.sibling = next.sibling
            bh.link(next, current)
        } else {
            if prev == nil {
                bh.head = next
            } else {
                prev.sibling = next
            }
            bh.link(current, next)
            current = next
        }
        next = current.sibling
    }
}

func (bh *BinomialHeap) merge(h1, h2 *BinomialNode) *BinomialNode {
    if h1 == nil {
        return h2
    }
    if h2 == nil {
        return h1
    }
    
    var result *BinomialNode
    var tail *BinomialNode
    
    for h1 != nil && h2 != nil {
        if h1.degree <= h2.degree {
            if result == nil {
                result = h1
                tail = h1
            } else {
                tail.sibling = h1
                tail = h1
            }
            h1 = h1.sibling
        } else {
            if result == nil {
                result = h2
                tail = h2
            } else {
                tail.sibling = h2
                tail = h2
            }
            h2 = h2.sibling
        }
    }
    
    if h1 != nil {
        tail.sibling = h1
    } else {
        tail.sibling = h2
    }
    
    return result
}

func (bh *BinomialHeap) link(child, parent *BinomialNode) {
    child.parent = parent
    child.sibling = parent.child
    parent.child = child
    parent.degree++
}

func (bh *BinomialHeap) reverseList(head *BinomialNode) *BinomialNode {
    var prev *BinomialNode
    current := head
    
    for current != nil {
        next := current.sibling
        current.sibling = prev
        current.parent = nil
        prev = current
        current = next
    }
    
    return prev
}
```

## Advanced Tries

### Suffix Tree

**Problem**: Implement a suffix tree for string matching.

```go
// Suffix Tree
type SuffixTreeNode struct {
    children map[byte]*SuffixTreeNode
    start    int
    end      int
    suffixLink *SuffixTreeNode
}

type SuffixTree struct {
    root *SuffixTreeNode
    text string
}

func NewSuffixTree(text string) *SuffixTree {
    st := &SuffixTree{
        root: &SuffixTreeNode{
            children: make(map[byte]*SuffixTreeNode),
            start:    -1,
            end:      -1,
        },
        text: text + "$", // Add sentinel
    }
    
    st.buildSuffixTree()
    return st
}

func (st *SuffixTree) buildSuffixTree() {
    n := len(st.text)
    activeNode := st.root
    activeEdge := -1
    activeLength := 0
    remainingSuffixes := 0
    
    for i := 0; i < n; i++ {
        remainingSuffixes++
        lastNewNode := (*SuffixTreeNode)(nil)
        
        for remainingSuffixes > 0 {
            if activeLength == 0 {
                activeEdge = i
            }
            
            if _, exists := activeNode.children[st.text[activeEdge]]; !exists {
                // Rule 2: Create new leaf
                activeNode.children[st.text[activeEdge]] = &SuffixTreeNode{
                    children: make(map[byte]*SuffixTreeNode),
                    start:    i,
                    end:      n - 1,
                }
                
                if lastNewNode != nil {
                    lastNewNode.suffixLink = activeNode
                    lastNewNode = nil
                }
            } else {
                nextNode := activeNode.children[st.text[activeEdge]]
                edgeLength := nextNode.end - nextNode.start + 1
                
                if activeLength >= edgeLength {
                    activeLength -= edgeLength
                    activeEdge += edgeLength
                    activeNode = nextNode
                    continue
                }
                
                if st.text[nextNode.start+activeLength] == st.text[i] {
                    // Rule 3: Extend
                    if lastNewNode != nil && activeNode != st.root {
                        lastNewNode.suffixLink = activeNode
                        lastNewNode = nil
                    }
                    activeLength++
                    break
                }
                
                // Rule 2: Split
                splitEnd := nextNode.start + activeLength - 1
                splitNode := &SuffixTreeNode{
                    children: make(map[byte]*SuffixTreeNode),
                    start:    nextNode.start,
                    end:      splitEnd,
                }
                
                activeNode.children[st.text[activeEdge]] = splitNode
                nextNode.start += activeLength
                splitNode.children[st.text[nextNode.start]] = nextNode
                
                // Create new leaf
                splitNode.children[st.text[i]] = &SuffixTreeNode{
                    children: make(map[byte]*SuffixTreeNode),
                    start:    i,
                    end:      n - 1,
                }
                
                if lastNewNode != nil {
                    lastNewNode.suffixLink = splitNode
                }
                lastNewNode = splitNode
            }
            
            remainingSuffixes--
            
            if activeNode == st.root && activeLength > 0 {
                activeLength--
                activeEdge = i - remainingSuffixes + 1
            } else if activeNode != st.root {
                activeNode = activeNode.suffixLink
            }
        }
    }
}

func (st *SuffixTree) Search(pattern string) bool {
    node := st.root
    i := 0
    
    for i < len(pattern) {
        if _, exists := node.children[pattern[i]]; !exists {
            return false
        }
        
        node = node.children[pattern[i]]
        j := node.start
        
        for j <= node.end && i < len(pattern) {
            if st.text[j] != pattern[i] {
                return false
            }
            i++
            j++
        }
    }
    
    return true
}
```

## Advanced Segment Trees

### Lazy Propagation Segment Tree

**Problem**: Implement a segment tree with lazy propagation for range updates.

```go
// Lazy Propagation Segment Tree
type LazySegmentTree struct {
    tree   []int
    lazy   []int
    size   int
    arr    []int
}

func NewLazySegmentTree(arr []int) *LazySegmentTree {
    n := len(arr)
    size := 1
    for size < n {
        size <<= 1
    }
    
    lst := &LazySegmentTree{
        tree: make([]int, 2*size),
        lazy: make([]int, 2*size),
        size: size,
        arr:  arr,
    }
    
    lst.build(0, 0, size-1)
    return lst
}

func (lst *LazySegmentTree) build(node, start, end int) {
    if start == end {
        if start < len(lst.arr) {
            lst.tree[node] = lst.arr[start]
        }
        return
    }
    
    mid := (start + end) / 2
    lst.build(2*node+1, start, mid)
    lst.build(2*node+2, mid+1, end)
    lst.tree[node] = lst.tree[2*node+1] + lst.tree[2*node+2]
}

func (lst *LazySegmentTree) updateRange(l, r, val int) {
    lst.updateRangeHelper(0, 0, lst.size-1, l, r, val)
}

func (lst *LazySegmentTree) updateRangeHelper(node, start, end, l, r, val int) {
    // Apply lazy updates
    if lst.lazy[node] != 0 {
        lst.tree[node] += lst.lazy[node] * (end - start + 1)
        if start != end {
            lst.lazy[2*node+1] += lst.lazy[node]
            lst.lazy[2*node+2] += lst.lazy[node]
        }
        lst.lazy[node] = 0
    }
    
    // No overlap
    if start > r || end < l {
        return
    }
    
    // Complete overlap
    if start >= l && end <= r {
        lst.tree[node] += val * (end - start + 1)
        if start != end {
            lst.lazy[2*node+1] += val
            lst.lazy[2*node+2] += val
        }
        return
    }
    
    // Partial overlap
    mid := (start + end) / 2
    lst.updateRangeHelper(2*node+1, start, mid, l, r, val)
    lst.updateRangeHelper(2*node+2, mid+1, end, l, r, val)
    lst.tree[node] = lst.tree[2*node+1] + lst.tree[2*node+2]
}

func (lst *LazySegmentTree) queryRange(l, r int) int {
    return lst.queryRangeHelper(0, 0, lst.size-1, l, r)
}

func (lst *LazySegmentTree) queryRangeHelper(node, start, end, l, r int) int {
    // Apply lazy updates
    if lst.lazy[node] != 0 {
        lst.tree[node] += lst.lazy[node] * (end - start + 1)
        if start != end {
            lst.lazy[2*node+1] += lst.lazy[node]
            lst.lazy[2*node+2] += lst.lazy[node]
        }
        lst.lazy[node] = 0
    }
    
    // No overlap
    if start > r || end < l {
        return 0
    }
    
    // Complete overlap
    if start >= l && end <= r {
        return lst.tree[node]
    }
    
    // Partial overlap
    mid := (start + end) / 2
    left := lst.queryRangeHelper(2*node+1, start, mid, l, r)
    right := lst.queryRangeHelper(2*node+2, mid+1, end, l, r)
    return left + right
}
```

## Advanced Fenwick Trees

### 2D Fenwick Tree

**Problem**: Implement a 2D Fenwick tree for 2D range queries.

```go
// 2D Fenwick Tree
type FenwickTree2D struct {
    tree [][]int
    rows int
    cols int
}

func NewFenwickTree2D(rows, cols int) *FenwickTree2D {
    return &FenwickTree2D{
        tree: make([][]int, rows+1),
        rows: rows,
        cols: cols,
    }
}

func (ft *FenwickTree2D) init() {
    for i := 0; i <= ft.rows; i++ {
        ft.tree[i] = make([]int, ft.cols+1)
    }
}

func (ft *FenwickTree2D) update(row, col, val int) {
    for i := row; i <= ft.rows; i += i & (-i) {
        for j := col; j <= ft.cols; j += j & (-j) {
            ft.tree[i][j] += val
        }
    }
}

func (ft *FenwickTree2D) query(row, col int) int {
    sum := 0
    for i := row; i > 0; i -= i & (-i) {
        for j := col; j > 0; j -= j & (-j) {
            sum += ft.tree[i][j]
        }
    }
    return sum
}

func (ft *FenwickTree2D) rangeQuery(row1, col1, row2, col2 int) int {
    return ft.query(row2, col2) - ft.query(row1-1, col2) - 
           ft.query(row2, col1-1) + ft.query(row1-1, col1-1)
}
```

## Advanced Disjoint Sets

### Weighted Union-Find with Path Compression

**Problem**: Implement weighted union-find with path compression and size tracking.

```go
// Weighted Union-Find with Path Compression
type WeightedUnionFind struct {
    parent []int
    size   []int
    count  int
}

func NewWeightedUnionFind(n int) *WeightedUnionFind {
    parent := make([]int, n)
    size := make([]int, n)
    for i := 0; i < n; i++ {
        parent[i] = i
        size[i] = 1
    }
    
    return &WeightedUnionFind{
        parent: parent,
        size:   size,
        count:  n,
    }
}

func (wuf *WeightedUnionFind) Find(x int) int {
    if wuf.parent[x] != x {
        wuf.parent[x] = wuf.Find(wuf.parent[x])
    }
    return wuf.parent[x]
}

func (wuf *WeightedUnionFind) Union(x, y int) {
    rootX := wuf.Find(x)
    rootY := wuf.Find(y)
    
    if rootX == rootY {
        return
    }
    
    // Union by size
    if wuf.size[rootX] < wuf.size[rootY] {
        rootX, rootY = rootY, rootX
    }
    
    wuf.parent[rootY] = rootX
    wuf.size[rootX] += wuf.size[rootY]
    wuf.count--
}

func (wuf *WeightedUnionFind) Connected(x, y int) bool {
    return wuf.Find(x) == wuf.Find(y)
}

func (wuf *WeightedUnionFind) Count() int {
    return wuf.count
}

func (wuf *WeightedUnionFind) Size(x int) int {
    return wuf.size[wuf.Find(x)]
}
```

## Conclusion

Advanced data structure interviews test:

1. **Implementation Skills**: Ability to implement complex data structures
2. **Time Complexity**: Understanding of operations and their costs
3. **Space Complexity**: Memory usage optimization
4. **Real-world Applications**: When and how to use each structure
5. **Trade-offs**: Understanding advantages and disadvantages
6. **Optimization**: Techniques for improving performance
7. **Edge Cases**: Handling all possible scenarios

Mastering these advanced data structures demonstrates your readiness for senior engineering roles and complex system design challenges.

## Additional Resources

- [Advanced Data Structures](https://www.advanceddatastructures.com/)
- [Tree Algorithms](https://www.treealgorithms.com/)
- [Graph Data Structures](https://www.graphdatastructures.com/)
- [Hash Table Implementations](https://www.hashtableimplementations.com/)
- [Heap Data Structures](https://www.heapdatastructures.com/)
- [Trie Implementations](https://www.trieimplementations.com/)
- [Segment Tree Algorithms](https://www.segmenttreealgorithms.com/)
- [Fenwick Tree Applications](https://www.fenwicktreeapplications.com/)
- [Disjoint Set Algorithms](https://www.disjointsetalgorithms.com/)
- [Data Structure Design](https://www.datastructuredesign.com/)
