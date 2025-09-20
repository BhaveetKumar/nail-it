# Arrays Pattern

> **Master array manipulation techniques with Go implementations**

## 📋 Problems

### **Two Pointers**

- [Two Sum](TwoSum.md) - Find two numbers that add up to target
- [Container With Most Water](ContainerWithMostWater.md) - Maximum area between two lines
- [3Sum](3Sum.md) - Find all unique triplets that sum to zero
- [4Sum](4Sum.md) - Find all unique quadruplets that sum to target
- [Remove Duplicates](RemoveDuplicates.md) - Remove duplicates from sorted array
- [Move Zeroes](MoveZeroes.md) - Move all zeros to end

### **Sliding Window**

- [Maximum Subarray](MaximumSubarray.md) - Kadane's algorithm
- [Longest Substring Without Repeating Characters](LongestSubstring.md) - Sliding window technique
- [Minimum Window Substring](../SlidingWindow/MinimumWindowSubstring.md) - Variable window size
- [Longest Repeating Character Replacement](../SlidingWindow/LongestRepeatingCharacterReplacement.md) - Sliding window with character replacement

### **Prefix Sum**

- [Subarray Sum Equals K](SubarraySumEqualsK.md) - Count subarrays with sum K
- [Range Sum Query](RangeSumQuery.md) - Multiple range sum queries
- [Product of Array Except Self](ProductOfArrayExceptSelf.md) - Array without division

### **Matrix Operations**

- [Spiral Matrix](SpiralMatrix.md) - Traverse matrix in spiral order
- [Rotate Image](RotateImage.md) - Rotate matrix 90 degrees clockwise
- [Set Matrix Zeroes](SetMatrixZeroes.md) - Set entire row/column to zero

### **Sorting & Searching**

- [Merge Sorted Arrays](MergeSortedArrays.md) - Merge two sorted arrays
- [Find First and Last Position](FindFirstAndLastPosition.md) - Binary search for range
- [Search in Rotated Sorted Array](SearchInRotatedSortedArray.md) - Binary search in rotated array

---

## 🎯 Key Concepts

### **Two Pointers Technique**

**Detailed Explanation:**
The two pointers technique is a fundamental algorithmic pattern that uses two pointers to traverse an array or sequence from different positions. This technique is particularly effective for problems involving sorted arrays, palindromes, or finding pairs/triplets that satisfy certain conditions.

**Why Two Pointers Work:**

- **Efficiency**: Reduces time complexity from O(n²) to O(n) for many problems
- **Space Optimization**: Uses O(1) extra space instead of O(n) for hash maps
- **Intuitive Logic**: Mimics human problem-solving approach of comparing elements from both ends

**Core Principles:**

1. **Sorted Array Advantage**: When array is sorted, we can make intelligent decisions about which pointer to move
2. **Condition-Based Movement**: Move pointers based on the relationship between current elements and target
3. **Convergence**: Pointers move toward each other, ensuring we don't miss any valid combinations

**Go Implementation Pattern:**

```go
func twoPointers(nums []int, target int) []int {
    left, right := 0, len(nums)-1

    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    return []int{-1, -1}
}
```

**Discussion Questions & Answers:**

**Q1: When should you use the two pointers technique instead of a hash map?**

**Answer:** Use two pointers when:

- **Sorted Array**: The array is already sorted or can be sorted
- **Space Constraints**: You need O(1) space complexity
- **Multiple Targets**: Finding all pairs/triplets, not just one
- **Range Queries**: Working with subarrays or contiguous elements
- **Performance Critical**: When O(n) time complexity is essential

**Q2: How do you handle duplicates in two pointers problems?**

**Answer:** Several strategies for handling duplicates:

- **Skip Duplicates**: After finding a valid pair, skip all duplicate elements
- **Hash Set**: Use a set to track already processed combinations
- **Index Tracking**: Keep track of processed indices to avoid duplicates
- **Sorting**: Sort the array first to group duplicates together

**Q3: What are the limitations of the two pointers technique?**

**Answer:** Key limitations include:

- **Sorted Requirement**: Works best with sorted arrays
- **Linear Movement**: Pointers can only move in one direction
- **Limited Patterns**: Not suitable for complex relationship patterns
- **Memory Access**: May have poor cache locality with large arrays

### **Sliding Window**

**Detailed Explanation:**
The sliding window technique is a method for efficiently processing contiguous subarrays or substrings. It maintains a "window" of elements and slides it across the data structure, updating the window's state as it moves.

**Why Sliding Window Works:**

- **Avoids Recalculation**: Reuses previous calculations instead of recalculating from scratch
- **Optimal Time Complexity**: Achieves O(n) time for many problems that would otherwise be O(n²)
- **Memory Efficient**: Uses O(1) or O(k) space where k is the window size

**Window Types:**

1. **Fixed Size Window**: Window size remains constant
2. **Variable Size Window**: Window size changes based on conditions
3. **Dynamic Window**: Window size adapts to meet specific criteria

**Go Implementation Pattern:**

```go
func slidingWindow(nums []int, k int) int {
    windowSum := 0
    maxSum := math.MinInt32

    // Initialize first window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    maxSum = windowSum

    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }

    return maxSum
}
```

**Discussion Questions & Answers:**

**Q1: How do you choose between fixed and variable window sizes?**

**Answer:** Choose based on the problem requirements:

- **Fixed Window**: When you need to process exactly k elements at a time
- **Variable Window**: When you need to find the optimal window size
- **Dynamic Window**: When window size depends on the content or conditions
- **Consider Trade-offs**: Fixed windows are simpler but less flexible

**Q2: What are the common pitfalls when implementing sliding window?**

**Answer:** Common pitfalls include:

- **Off-by-One Errors**: Incorrect boundary conditions
- **State Management**: Not properly updating window state
- **Edge Cases**: Empty arrays, single elements, or invalid inputs
- **Memory Leaks**: Not properly cleaning up window state
- **Index Management**: Confusion between 0-based and 1-based indexing

**Q3: How do you optimize sliding window for very large datasets?**

**Answer:** Optimization strategies:

- **Lazy Evaluation**: Only compute when needed
- **Batch Processing**: Process multiple windows simultaneously
- **Memory Pooling**: Reuse data structures to reduce allocations
- **Parallel Processing**: Use goroutines for independent windows
- **Streaming**: Process data as it arrives instead of loading all at once

### **Prefix Sum**

**Detailed Explanation:**
Prefix sum (also known as cumulative sum) is a technique that precomputes the sum of elements from the beginning of the array up to each position. This allows for O(1) range sum queries after O(n) preprocessing.

**Why Prefix Sum Works:**

- **Mathematical Property**: Sum of range [i, j] = prefix[j] - prefix[i-1]
- **Preprocessing Benefit**: One-time O(n) cost for multiple O(1) queries
- **Space-Time Trade-off**: Uses O(n) extra space for O(1) query time

**Core Formula:**

```
prefix[i] = sum of elements from index 0 to i
rangeSum(i, j) = prefix[j] - prefix[i-1] (when i > 0)
rangeSum(0, j) = prefix[j]
```

**Go Implementation Pattern:**

```go
func prefixSum(nums []int) []int {
    prefix := make([]int, len(nums)+1)
    for i := 0; i < len(nums); i++ {
        prefix[i+1] = prefix[i] + nums[i]
    }
    return prefix
}

func rangeSum(prefix []int, left, right int) int {
    return prefix[right+1] - prefix[left]
}
```

**Discussion Questions & Answers:**

**Q1: When is prefix sum more efficient than calculating sums on demand?**

**Answer:** Prefix sum is more efficient when:

- **Multiple Queries**: You need to answer many range sum queries
- **Query Frequency**: More than O(n) queries on an array of size n
- **Real-time Systems**: When query response time is critical
- **Batch Processing**: When you can preprocess data once and query multiple times
- **Memory Available**: When you have sufficient memory for the prefix array

**Q2: How do you handle updates to the original array with prefix sum?**

**Answer:** Several approaches for handling updates:

- **Rebuild**: Recompute prefix array after each update (O(n) per update)
- **Fenwick Tree**: Use binary indexed tree for O(log n) updates and queries
- **Segment Tree**: Use segment tree for range updates and queries
- **Lazy Propagation**: Defer updates until queries are made
- **Batch Updates**: Accumulate updates and apply them together

**Q3: What are the space and time trade-offs of prefix sum?**

**Answer:** Trade-offs include:

- **Space**: O(n) extra space for prefix array
- **Time**: O(n) preprocessing, O(1) per query
- **Updates**: O(n) per update if rebuilding, O(log n) with advanced data structures
- **Memory Access**: Good cache locality for sequential access
- **Scalability**: May not be suitable for very large arrays due to memory constraints

---

## 🛠️ Go-Specific Tips

### **Array Initialization**

**Detailed Explanation:**
Go provides several ways to initialize arrays and slices, each with different performance characteristics and use cases. Understanding these differences is crucial for writing efficient Go code.

**Array vs Slice:**

- **Array**: Fixed-size collection with value semantics
- **Slice**: Dynamic-size collection with reference semantics
- **Performance**: Arrays are generally faster but less flexible

**Go Implementation Patterns:**

```go
// Fixed size array - allocated on stack
arr := [5]int{1, 2, 3, 4, 5}

// Dynamic slice with pre-allocated capacity
slice := make([]int, 0, n)  // length 0, capacity n
slice := make([]int, n)     // length n, capacity n

// Slice from array - shares underlying memory
slice := arr[1:4]  // elements from index 1 to 3

// Slice literal - allocated on heap
slice := []int{1, 2, 3, 4, 5}
```

**Discussion Questions & Answers:**

**Q1: When should you use arrays vs slices in Go?**

**Answer:** Use arrays when:

- **Fixed Size**: You know the exact size at compile time
- **Performance Critical**: Arrays are slightly faster due to stack allocation
- **Value Semantics**: You want to pass by value without reference sharing
- **Memory Layout**: You need predictable memory layout for optimization

Use slices when:

- **Dynamic Size**: Size is determined at runtime
- **Flexibility**: You need to grow or shrink the collection
- **Reference Semantics**: You want to share data between functions
- **Standard Practice**: Most Go code uses slices for collections

**Q2: How do you optimize slice allocation in Go?**

**Answer:** Optimization strategies:

- **Pre-allocate Capacity**: Use `make([]int, 0, expectedSize)` when you know the approximate size
- **Reuse Slices**: Reset length with `slice = slice[:0]` to reuse capacity
- **Avoid Unnecessary Allocations**: Use slice literals only when necessary
- **Pool Slices**: Use `sync.Pool` for frequently allocated slices
- **Batch Operations**: Minimize individual append operations

### **Common Operations**

**Detailed Explanation:**
Go provides efficient built-in operations for slice manipulation. Understanding the performance characteristics of these operations is essential for writing efficient code.

**Operation Categories:**

1. **Modification Operations**: Append, copy, sort
2. **Access Operations**: Indexing, slicing, iteration
3. **Utility Operations**: Length, capacity, comparison

**Go Implementation Patterns:**

```go
// Append elements - may cause reallocation
slice = append(slice, element)
slice = append(slice, elements...)

// Copy slice - creates independent copy
newSlice := make([]int, len(slice))
copy(newSlice, slice)

// Sort slice - in-place sorting
sort.Ints(slice)
sort.Slice(slice, func(i, j int) bool {
    return slice[i] < slice[j]
})

// Efficient iteration
for i, v := range slice {
    // Process element at index i with value v
}
```

**Discussion Questions & Answers:**

**Q1: What are the performance implications of different slice operations?**

**Answer:** Performance characteristics:

- **Append**: O(1) amortized, O(n) worst case due to reallocation
- **Copy**: O(n) time complexity, creates new slice
- **Sort**: O(n log n) time complexity, in-place operation
- **Indexing**: O(1) time complexity, direct memory access
- **Slicing**: O(1) time complexity, creates new slice header

**Q2: How do you avoid common slice-related bugs in Go?**

**Answer:** Common bugs and prevention:

- **Index Out of Bounds**: Always check bounds before accessing
- **Slice Sharing**: Be aware that slices share underlying arrays
- **Capacity Issues**: Monitor capacity vs length to avoid reallocations
- **Memory Leaks**: Avoid holding references to large slices unnecessarily
- **Concurrent Access**: Use proper synchronization for concurrent slice access

### **Memory Optimization**

**Detailed Explanation:**
Memory optimization in Go involves understanding how slices are allocated and managed. Proper memory management can significantly improve performance and reduce garbage collection pressure.

**Memory Management Concepts:**

- **Stack vs Heap**: Understanding where data is allocated
- **Garbage Collection**: Minimizing GC pressure through efficient allocation
- **Memory Pooling**: Reusing allocated memory to reduce allocations

**Go Implementation Patterns:**

```go
// Pre-allocate capacity when size is known
result := make([]int, 0, expectedSize)

// Reuse slice to avoid allocations
slice = slice[:0]  // reset length to 0, keep capacity

// Memory pooling for high-frequency allocations
var slicePool = sync.Pool{
    New: func() interface{} {
        return make([]int, 0, 1000)
    },
}

func getSlice() []int {
    return slicePool.Get().([]int)
}

func putSlice(slice []int) {
    slice = slice[:0]  // reset length
    slicePool.Put(slice)
}
```

**Discussion Questions & Answers:**

**Q1: How do you minimize garbage collection pressure with slices?**

**Answer:** Strategies to reduce GC pressure:

- **Pre-allocate Capacity**: Avoid frequent reallocations
- **Reuse Slices**: Reset length instead of creating new slices
- **Memory Pooling**: Use `sync.Pool` for frequently allocated slices
- **Batch Operations**: Minimize individual allocations
- **Avoid Unnecessary Copies**: Use slice references when possible

**Q2: What are the trade-offs between different memory optimization techniques?**

**Answer:** Trade-offs include:

- **Memory vs Performance**: More memory usage for better performance
- **Complexity vs Efficiency**: Simpler code vs optimized code
- **Cache Locality**: Stack allocation vs heap allocation
- **Concurrency**: Thread-safe operations vs performance
- **Maintainability**: Optimized code vs readable code
