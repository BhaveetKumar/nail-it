# Arrays Pattern

> **Master array manipulation techniques with Go implementations**

## 📋 Problems

### **Two Pointers**
- [Two Sum](./TwoSum.md) - Find two numbers that add up to target
- [Container With Most Water](./ContainerWithMostWater.md) - Maximum area between two lines
- [3Sum](./3Sum.md) - Find all unique triplets that sum to zero
- [4Sum](./4Sum.md) - Find all unique quadruplets that sum to target
- [Remove Duplicates](./RemoveDuplicates.md) - Remove duplicates from sorted array
- [Move Zeroes](./MoveZeroes.md) - Move all zeros to end

### **Sliding Window**
- [Maximum Subarray](./MaximumSubarray.md) - Kadane's algorithm
- [Longest Substring Without Repeating Characters](./LongestSubstring.md) - Sliding window technique
- [Minimum Window Substring](./MinimumWindowSubstring.md) - Variable window size
- [Longest Repeating Character Replacement](./LongestRepeatingCharacterReplacement.md) - Sliding window with character replacement

### **Prefix Sum**
- [Subarray Sum Equals K](./SubarraySumEqualsK.md) - Count subarrays with sum K
- [Range Sum Query](./RangeSumQuery.md) - Multiple range sum queries
- [Product of Array Except Self](./ProductOfArrayExceptSelf.md) - Array without division

### **Matrix Operations**
- [Spiral Matrix](./SpiralMatrix.md) - Traverse matrix in spiral order
- [Rotate Image](./RotateImage.md) - Rotate matrix 90 degrees clockwise
- [Set Matrix Zeroes](./SetMatrixZeroes.md) - Set entire row/column to zero

### **Sorting & Searching**
- [Merge Sorted Arrays](./MergeSortedArrays.md) - Merge two sorted arrays
- [Find First and Last Position](./FindFirstAndLastPosition.md) - Binary search for range
- [Search in Rotated Sorted Array](./SearchInRotatedSortedArray.md) - Binary search in rotated array

---

## 🎯 Key Concepts

### **Two Pointers Technique**
- **Use Case**: When you need to find pairs or triplets in sorted arrays
- **Time Complexity**: O(n) instead of O(n²)
- **Go Implementation**: Use left and right pointers

### **Sliding Window**
- **Fixed Window**: Window size is constant
- **Variable Window**: Window size changes based on condition
- **Go Implementation**: Use start and end pointers

### **Prefix Sum**
- **Use Case**: Multiple range sum queries
- **Time Complexity**: O(1) per query after O(n) preprocessing
- **Go Implementation**: Precompute prefix sums

---

## 🛠️ Go-Specific Tips

### **Array Initialization**
```go
// Fixed size array
arr := [5]int{1, 2, 3, 4, 5}

// Dynamic slice
slice := make([]int, 0, n)  // length 0, capacity n
slice := make([]int, n)     // length n, capacity n

// Slice from array
slice := arr[1:4]  // elements from index 1 to 3
```

### **Common Operations**
```go
// Append elements
slice = append(slice, element)

// Copy slice
newSlice := make([]int, len(slice))
copy(newSlice, slice)

// Sort slice
sort.Ints(slice)
sort.Slice(slice, func(i, j int) bool {
    return slice[i] < slice[j]
})
```

### **Memory Optimization**
```go
// Pre-allocate capacity when size is known
result := make([]int, 0, expectedSize)

// Reuse slice to avoid allocations
slice = slice[:0]  // reset length to 0, keep capacity
```
