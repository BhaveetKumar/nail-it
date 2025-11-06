---
# Auto-generated front matter
Title: Segment Tree
LastUpdated: 2025-11-06T20:45:58.681311
Tags: []
Status: draft
---

# Segment Tree - Advanced Data Structure for Range Queries

## Overview

A Segment Tree is a tree data structure that allows efficient range queries and range updates on an array. It's particularly useful for problems involving range sum, range minimum/maximum, and range updates.

## Key Concepts

- **Leaf Nodes**: Represent individual array elements
- **Internal Nodes**: Represent ranges and store computed values
- **Range Queries**: Query operations over a range [L, R]
- **Range Updates**: Update operations over a range [L, R]
- **Lazy Propagation**: Optimize range updates by deferring updates

## Segment Tree Structure

```mermaid
graph TD
    A[Root: [0,7]] --> B[Left: [0,3]]
    A --> C[Right: [4,7]]
    
    B --> D[Left: [0,1]]
    B --> E[Right: [2,3]]
    
    C --> F[Left: [4,5]]
    C --> G[Right: [6,7]]
    
    D --> H[Leaf: [0,0]]
    D --> I[Leaf: [1,1]]
    E --> J[Leaf: [2,2]]
    E --> K[Leaf: [3,3]]
    F --> L[Leaf: [4,4]]
    F --> M[Leaf: [5,5]]
    G --> N[Leaf: [6,6]]
    G --> O[Leaf: [7,7]]
```

## Go Implementation

```go
package main

import (
    "fmt"
    "math"
)

// SegmentTree represents a segment tree
type SegmentTree struct {
    tree   []int
    lazy   []int
    size   int
    data   []int
}

// NewSegmentTree creates a new segment tree
func NewSegmentTree(data []int) *SegmentTree {
    n := len(data)
    size := 1
    for size < n {
        size <<= 1
    }
    
    st := &SegmentTree{
        tree: make([]int, 2*size),
        lazy: make([]int, 2*size),
        size: size,
        data: data,
    }
    
    st.build(0, 0, n-1)
    return st
}

// build builds the segment tree
func (st *SegmentTree) build(node, start, end int) {
    if start == end {
        st.tree[node] = st.data[start]
        return
    }
    
    mid := (start + end) / 2
    st.build(2*node+1, start, mid)
    st.build(2*node+2, mid+1, end)
    st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
}

// QueryRangeSum queries the sum in range [l, r]
func (st *SegmentTree) QueryRangeSum(l, r int) int {
    return st.queryRangeSum(0, 0, st.size-1, l, r)
}

// queryRangeSum helper function for range sum query
func (st *SegmentTree) queryRangeSum(node, start, end, l, r int) int {
    // Lazy propagation
    if st.lazy[node] != 0 {
        st.tree[node] += st.lazy[node] * (end - start + 1)
        if start != end {
            st.lazy[2*node+1] += st.lazy[node]
            st.lazy[2*node+2] += st.lazy[node]
        }
        st.lazy[node] = 0
    }
    
    // No overlap
    if start > r || end < l {
        return 0
    }
    
    // Complete overlap
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    // Partial overlap
    mid := (start + end) / 2
    leftSum := st.queryRangeSum(2*node+1, start, mid, l, r)
    rightSum := st.queryRangeSum(2*node+2, mid+1, end, l, r)
    return leftSum + rightSum
}

// UpdateRange updates values in range [l, r] by adding val
func (st *SegmentTree) UpdateRange(l, r, val int) {
    st.updateRange(0, 0, st.size-1, l, r, val)
}

// updateRange helper function for range update
func (st *SegmentTree) updateRange(node, start, end, l, r, val int) {
    // Lazy propagation
    if st.lazy[node] != 0 {
        st.tree[node] += st.lazy[node] * (end - start + 1)
        if start != end {
            st.lazy[2*node+1] += st.lazy[node]
            st.lazy[2*node+2] += st.lazy[node]
        }
        st.lazy[node] = 0
    }
    
    // No overlap
    if start > r || end < l {
        return
    }
    
    // Complete overlap
    if l <= start && end <= r {
        st.tree[node] += val * (end - start + 1)
        if start != end {
            st.lazy[2*node+1] += val
            st.lazy[2*node+2] += val
        }
        return
    }
    
    // Partial overlap
    mid := (start + end) / 2
    st.updateRange(2*node+1, start, mid, l, r, val)
    st.updateRange(2*node+2, mid+1, end, l, r, val)
    st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
}

// UpdatePoint updates a single point
func (st *SegmentTree) UpdatePoint(index, val int) {
    st.updatePoint(0, 0, st.size-1, index, val)
}

// updatePoint helper function for point update
func (st *SegmentTree) updatePoint(node, start, end, index, val int) {
    if start == end {
        st.tree[node] = val
        return
    }
    
    mid := (start + end) / 2
    if index <= mid {
        st.updatePoint(2*node+1, start, mid, index, val)
    } else {
        st.updatePoint(2*node+2, mid+1, end, index, val)
    }
    st.tree[node] = st.tree[2*node+1] + st.tree[2*node+2]
}

// QueryRangeMin queries the minimum in range [l, r]
func (st *SegmentTree) QueryRangeMin(l, r int) int {
    return st.queryRangeMin(0, 0, st.size-1, l, r)
}

// queryRangeMin helper function for range minimum query
func (st *SegmentTree) queryRangeMin(node, start, end, l, r int) int {
    // No overlap
    if start > r || end < l {
        return math.MaxInt32
    }
    
    // Complete overlap
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    // Partial overlap
    mid := (start + end) / 2
    leftMin := st.queryRangeMin(2*node+1, start, mid, l, r)
    rightMin := st.queryRangeMin(2*node+2, mid+1, end, l, r)
    return min(leftMin, rightMin)
}

// min returns the minimum of two integers
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// QueryRangeMax queries the maximum in range [l, r]
func (st *SegmentTree) QueryRangeMax(l, r int) int {
    return st.queryRangeMax(0, 0, st.size-1, l, r)
}

// queryRangeMax helper function for range maximum query
func (st *SegmentTree) queryRangeMax(node, start, end, l, r int) int {
    // No overlap
    if start > r || end < l {
        return math.MinInt32
    }
    
    // Complete overlap
    if l <= start && end <= r {
        return st.tree[node]
    }
    
    // Partial overlap
    mid := (start + end) / 2
    leftMax := st.queryRangeMax(2*node+1, start, mid, l, r)
    rightMax := st.queryRangeMax(2*node+2, mid+1, end, l, r)
    return max(leftMax, rightMax)
}

// max returns the maximum of two integers
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// PrintTree prints the segment tree
func (st *SegmentTree) PrintTree() {
    fmt.Println("Segment Tree:")
    st.printTree(0, 0, st.size-1, 0)
}

// printTree helper function to print the tree
func (st *SegmentTree) printTree(node, start, end, level int) {
    if start == end {
        for i := 0; i < level; i++ {
            fmt.Print("  ")
        }
        fmt.Printf("Leaf [%d,%d]: %d\n", start, end, st.tree[node])
        return
    }
    
    for i := 0; i < level; i++ {
        fmt.Print("  ")
    }
    fmt.Printf("Node [%d,%d]: %d\n", start, end, st.tree[node])
    
    mid := (start + end) / 2
    st.printTree(2*node+1, start, mid, level+1)
    st.printTree(2*node+2, mid+1, end, level+1)
}

// Example usage
func main() {
    data := []int{1, 3, 5, 7, 9, 11}
    st := NewSegmentTree(data)
    
    fmt.Println("Original array:", data)
    st.PrintTree()
    
    // Range sum query
    sum := st.QueryRangeSum(1, 4)
    fmt.Printf("Sum from index 1 to 4: %d\n", sum)
    
    // Range update
    st.UpdateRange(1, 3, 2)
    fmt.Println("After adding 2 to range [1,3]:")
    st.PrintTree()
    
    // Point update
    st.UpdatePoint(2, 10)
    fmt.Println("After updating index 2 to 10:")
    st.PrintTree()
    
    // Range minimum query
    min := st.QueryRangeMin(0, 5)
    fmt.Printf("Minimum in range [0,5]: %d\n", min)
    
    // Range maximum query
    max := st.QueryRangeMax(0, 5)
    fmt.Printf("Maximum in range [0,5]: %d\n", max)
}
```

## Applications

1. **Range Sum Queries**: Find sum of elements in a range
2. **Range Minimum/Maximum**: Find min/max in a range
3. **Range Updates**: Update all elements in a range
4. **Frequency Counting**: Count occurrences in ranges
5. **Inversion Count**: Count inversions in arrays

## Time Complexity

- **Build**: O(n)
- **Query**: O(log n)
- **Update**: O(log n)
- **Range Update**: O(log n) with lazy propagation

## Space Complexity

- **Storage**: O(n)

## Advantages

1. **Efficient Range Queries**: O(log n) time complexity
2. **Range Updates**: Support for range updates
3. **Flexible**: Can be adapted for different operations
4. **Lazy Propagation**: Optimizes range updates

## Disadvantages

1. **Memory Usage**: Requires 4n space
2. **Complexity**: More complex than simple arrays
3. **Overhead**: For small arrays, overhead might be significant

## Common Problems

1. **Range Sum Queries**: Sum of elements in range
2. **Range Minimum Queries**: Minimum element in range
3. **Range Maximum Queries**: Maximum element in range
4. **Range Updates**: Update all elements in range
5. **Frequency Queries**: Count elements in range

## Interview Questions

1. **How do you implement a segment tree?**
   - Build tree bottom-up, query/update recursively

2. **What is lazy propagation?**
   - Defer updates until necessary to optimize range updates

3. **When would you use a segment tree?**
   - When you need efficient range queries and updates

4. **How do you handle range updates efficiently?**
   - Use lazy propagation to defer updates

## Time Complexity Analysis

- **Build**: O(n) - Visit each element once
- **Query**: O(log n) - Height of tree is log n
- **Update**: O(log n) - Height of tree is log n
- **Range Update**: O(log n) - With lazy propagation

## Space Complexity Analysis

- **Tree Array**: O(4n) - 2n for tree, 2n for lazy
- **Recursion Stack**: O(log n) - Height of tree

The optimal solution uses:
1. **Efficient Building**: Build tree bottom-up
2. **Lazy Propagation**: Optimize range updates
3. **Proper Indexing**: Use 0-based indexing
4. **Memory Management**: Allocate exact space needed