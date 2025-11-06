---
# Auto-generated front matter
Title: Fenwick Tree
LastUpdated: 2025-11-06T20:45:58.683624
Tags: []
Status: draft
---

# Fenwick Tree (Binary Indexed Tree) - Efficient Range Queries

## Overview

A Fenwick Tree (also known as Binary Indexed Tree or BIT) is a data structure that provides efficient range sum queries and point updates. It's particularly useful for problems involving prefix sums and range queries.

## Key Concepts

- **Binary Representation**: Uses binary representation for efficient indexing
- **Prefix Sums**: Efficiently computes prefix sums
- **Point Updates**: Updates single elements efficiently
- **Range Queries**: Queries sum over ranges
- **Lowest Set Bit**: Uses the lowest set bit for navigation

## Fenwick Tree Structure

```mermaid
graph TD
    A[BIT[8]: Sum[1,8]] --> B[BIT[4]: Sum[1,4]]
    A --> C[BIT[6]: Sum[5,6]]
    A --> D[BIT[7]: Sum[7,7]]
    
    B --> E[BIT[2]: Sum[1,2]]
    B --> F[BIT[3]: Sum[3,3]]
    
    E --> G[BIT[1]: Sum[1,1]]
    E --> H[BIT[2]: Sum[2,2]]
    
    C --> I[BIT[5]: Sum[5,5]]
    C --> J[BIT[6]: Sum[6,6]]
```

## Go Implementation

```go
package main

import (
    "fmt"
    "math"
)

// FenwickTree represents a Fenwick Tree (Binary Indexed Tree)
type FenwickTree struct {
    tree []int
    size int
}

// NewFenwickTree creates a new Fenwick Tree
func NewFenwickTree(size int) *FenwickTree {
    return &FenwickTree{
        tree: make([]int, size+1),
        size: size,
    }
}

// NewFenwickTreeFromArray creates a Fenwick Tree from an array
func NewFenwickTreeFromArray(arr []int) *FenwickTree {
    ft := NewFenwickTree(len(arr))
    for i, val := range arr {
        ft.Update(i+1, val)
    }
    return ft
}

// Update updates the value at index i by adding val
func (ft *FenwickTree) Update(i, val int) {
    for i <= ft.size {
        ft.tree[i] += val
        i += i & (-i) // Add the lowest set bit
    }
}

// Query returns the prefix sum from 1 to i
func (ft *FenwickTree) Query(i int) int {
    sum := 0
    for i > 0 {
        sum += ft.tree[i]
        i -= i & (-i) // Remove the lowest set bit
    }
    return sum
}

// QueryRange returns the sum from index i to j (inclusive)
func (ft *FenwickTree) QueryRange(i, j int) int {
    if i > j {
        return 0
    }
    return ft.Query(j) - ft.Query(i-1)
}

// GetValue returns the value at index i
func (ft *FenwickTree) GetValue(i int) int {
    return ft.QueryRange(i, i)
}

// SetValue sets the value at index i to val
func (ft *FenwickTree) SetValue(i, val int) {
    current := ft.GetValue(i)
    ft.Update(i, val-current)
}

// RangeUpdate updates values from index i to j by adding val
func (ft *FenwickTree) RangeUpdate(i, j, val int) {
    ft.Update(i, val)
    ft.Update(j+1, -val)
}

// RangeQuery returns the sum from index i to j (inclusive)
func (ft *FenwickTree) RangeQuery(i, j int) int {
    return ft.QueryRange(i, j)
}

// PrintTree prints the Fenwick Tree
func (ft *FenwickTree) PrintTree() {
    fmt.Println("Fenwick Tree:")
    for i := 1; i <= ft.size; i++ {
        fmt.Printf("BIT[%d]: %d\n", i, ft.tree[i])
    }
}

// PrintArray prints the underlying array
func (ft *FenwickTree) PrintArray() {
    fmt.Println("Underlying Array:")
    for i := 1; i <= ft.size; i++ {
        fmt.Printf("arr[%d]: %d\n", i, ft.GetValue(i))
    }
}

// GetPrefixSums returns all prefix sums
func (ft *FenwickTree) GetPrefixSums() []int {
    prefixSums := make([]int, ft.size+1)
    for i := 1; i <= ft.size; i++ {
        prefixSums[i] = ft.Query(i)
    }
    return prefixSums
}

// GetRangeSums returns sum for all possible ranges
func (ft *FenwickTree) GetRangeSums() [][]int {
    rangeSums := make([][]int, ft.size+1)
    for i := 0; i <= ft.size; i++ {
        rangeSums[i] = make([]int, ft.size+1)
        for j := i; j <= ft.size; j++ {
            rangeSums[i][j] = ft.QueryRange(i, j)
        }
    }
    return rangeSums
}

// FindKthElement finds the k-th smallest element
func (ft *FenwickTree) FindKthElement(k int) int {
    left, right := 1, ft.size
    result := -1
    
    for left <= right {
        mid := (left + right) / 2
        if ft.Query(mid) >= k {
            result = mid
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    
    return result
}

// CountInversions counts inversions in the array
func (ft *FenwickTree) CountInversions(arr []int) int {
    // Coordinate compression
    compressed := make([]int, len(arr))
    copy(compressed, arr)
    
    // Sort and get unique values
    unique := make([]int, 0)
    seen := make(map[int]bool)
    for _, val := range arr {
        if !seen[val] {
            unique = append(unique, val)
            seen[val] = true
        }
    }
    
    // Sort unique values
    for i := 0; i < len(unique); i++ {
        for j := i + 1; j < len(unique); j++ {
            if unique[i] > unique[j] {
                unique[i], unique[j] = unique[j], unique[i]
            }
        }
    }
    
    // Create mapping
    mapping := make(map[int]int)
    for i, val := range unique {
        mapping[val] = i + 1
    }
    
    // Compress array
    for i, val := range arr {
        compressed[i] = mapping[val]
    }
    
    // Count inversions
    ft2 := NewFenwickTree(len(unique))
    inversions := 0
    
    for i := len(compressed) - 1; i >= 0; i-- {
        inversions += ft2.Query(compressed[i] - 1)
        ft2.Update(compressed[i], 1)
    }
    
    return inversions
}

// GetFrequency returns the frequency of each element
func (ft *FenwickTree) GetFrequency() []int {
    frequency := make([]int, ft.size+1)
    for i := 1; i <= ft.size; i++ {
        frequency[i] = ft.GetValue(i)
    }
    return frequency
}

// GetCumulativeFrequency returns cumulative frequency
func (ft *FenwickTree) GetCumulativeFrequency() []int {
    return ft.GetPrefixSums()
}

// Example usage
func main() {
    // Create Fenwick Tree
    ft := NewFenwickTree(8)
    
    // Update values
    values := []int{1, 3, 5, 7, 9, 11, 13, 15}
    for i, val := range values {
        ft.Update(i+1, val)
    }
    
    fmt.Println("Original array:", values)
    ft.PrintTree()
    ft.PrintArray()
    
    // Query operations
    fmt.Printf("Prefix sum from 1 to 4: %d\n", ft.Query(4))
    fmt.Printf("Range sum from 2 to 6: %d\n", ft.QueryRange(2, 6))
    fmt.Printf("Value at index 3: %d\n", ft.GetValue(3))
    
    // Update operations
    ft.SetValue(3, 10)
    fmt.Println("After setting index 3 to 10:")
    ft.PrintArray()
    
    // Range update
    ft.RangeUpdate(2, 5, 2)
    fmt.Println("After adding 2 to range [2,5]:")
    ft.PrintArray()
    
    // Prefix sums
    prefixSums := ft.GetPrefixSums()
    fmt.Println("Prefix sums:", prefixSums)
    
    // Range sums
    rangeSums := ft.GetRangeSums()
    fmt.Println("Range sums:")
    for i := 1; i <= ft.size; i++ {
        for j := i; j <= ft.size; j++ {
            if rangeSums[i][j] != 0 {
                fmt.Printf("Range [%d,%d]: %d\n", i, j, rangeSums[i][j])
            }
        }
    }
    
    // Count inversions
    arr := []int{5, 4, 3, 2, 1}
    inversions := ft.CountInversions(arr)
    fmt.Printf("Inversions in %v: %d\n", arr, inversions)
    
    // Find k-th element
    kth := ft.FindKthElement(10)
    fmt.Printf("10-th smallest element: %d\n", kth)
}
```

## Applications

1. **Range Sum Queries**: Efficient range sum queries
2. **Point Updates**: Update single elements
3. **Prefix Sums**: Compute prefix sums efficiently
4. **Inversion Counting**: Count inversions in arrays
5. **Frequency Arrays**: Maintain frequency counts

## Time Complexity

- **Update**: O(log n)
- **Query**: O(log n)
- **Range Query**: O(log n)
- **Range Update**: O(log n)

## Space Complexity

- **Storage**: O(n)

## Advantages

1. **Efficient**: O(log n) for both updates and queries
2. **Simple**: Easy to implement and understand
3. **Memory Efficient**: Uses only O(n) space
4. **Fast**: Very fast in practice

## Disadvantages

1. **Limited Operations**: Only supports sum operations
2. **1-based Indexing**: Uses 1-based indexing
3. **Not General**: Not suitable for all range operations

## Common Problems

1. **Range Sum Queries**: Sum of elements in range
2. **Point Updates**: Update single elements
3. **Inversion Count**: Count inversions in arrays
4. **Frequency Queries**: Count frequencies
5. **Prefix Sums**: Compute prefix sums

## Interview Questions

1. **How do you implement a Fenwick Tree?**
   - Use binary representation and lowest set bit

2. **What is the time complexity of Fenwick Tree?**
   - O(log n) for both updates and queries

3. **When would you use a Fenwick Tree?**
   - When you need efficient range sum queries and point updates

4. **How do you handle range updates?**
   - Use two point updates with opposite signs

## Time Complexity Analysis

- **Update**: O(log n) - Height of tree is log n
- **Query**: O(log n) - Height of tree is log n
- **Range Query**: O(log n) - Two queries
- **Range Update**: O(log n) - Two updates

## Space Complexity Analysis

- **Tree Array**: O(n) - One array of size n+1
- **Recursion Stack**: O(1) - No recursion used

## Comparison with Segment Tree

| Feature | Fenwick Tree | Segment Tree |
|---------|--------------|--------------|
| Space | O(n) | O(4n) |
| Update | O(log n) | O(log n) |
| Query | O(log n) | O(log n) |
| Range Update | O(log n) | O(log n) |
| Operations | Sum only | Any associative operation |
| Complexity | Simple | More complex |

The optimal solution uses:
1. **Efficient Indexing**: Use 1-based indexing
2. **Binary Operations**: Use bit manipulation for efficiency
3. **Memory Management**: Allocate exact space needed
4. **Error Handling**: Handle edge cases properly