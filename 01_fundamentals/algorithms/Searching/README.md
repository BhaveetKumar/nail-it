---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.706675
Tags: []
Status: draft
---

# Searching Pattern

> **Master searching algorithms and their applications with Go implementations**

## 📋 Problems

### **Binary Search**

- [Binary Search](BinarySearch.md) - Search in sorted array
- [Search Insert Position](SearchInsertPosition.md) - Find insertion point
- [Find First and Last Position](../Arrays/FindFirstAndLastPosition.md) - Search range
- [Search in Rotated Sorted Array](SearchInRotatedSortedArray.md) - Search in rotated array
- [Find Minimum in Rotated Sorted Array](../Arrays/FindMinimumInRotatedSortedArray.md) - Find minimum

### **Advanced Searching**

- [Search a 2D Matrix](Search2DMatrix.md) - Search in 2D matrix
- [Search in Rotated Sorted Array II](SearchInRotatedSortedArrayII.md) - With duplicates
- [Find Peak Element](FindPeakElement.md) - Find local maximum
- [Sqrt(x)](../Math/Sqrt.md) - Binary search for square root
- [Pow(x, n)](../Math/Pow.md) - Fast power calculation

### **Search Applications**

- [Guess Number Higher or Lower](GuessNumber.md) - Interactive search
- [First Bad Version](FirstBadVersion.md) - Find first occurrence
- [Kth Smallest Element](KthSmallestElement.md) - Find kth element
- [Median of Two Sorted Arrays](../Arrays/MedianOfTwoSortedArrays.md) - Find median
- [Search Suggestions System](SearchSuggestionsSystem.md) - Prefix search

---

## 🎯 Key Concepts

### **Search Algorithms**

**Detailed Explanation:**
Search algorithms are fundamental techniques for finding elements or values in data structures. The choice of search algorithm depends on the data structure, whether the data is sorted, and the specific requirements of the problem.

**1. Linear Search:**

- **Time Complexity**: O(n) - Check each element sequentially
- **Space Complexity**: O(1) - No extra space required
- **When to Use**: Unsorted arrays, small datasets, or when data is not sorted
- **Algorithm**: Iterate through each element until target is found
- **Advantages**: Simple to implement, works on any data structure
- **Disadvantages**: Inefficient for large datasets

**2. Binary Search:**

- **Time Complexity**: O(log n) - Divide and conquer approach
- **Space Complexity**: O(1) for iterative, O(log n) for recursive
- **When to Use**: Sorted arrays or monotonic functions
- **Algorithm**: Compare target with middle element, eliminate half the search space
- **Advantages**: Very efficient for large datasets, logarithmic time complexity
- **Disadvantages**: Requires sorted data, more complex to implement

**3. Ternary Search:**

- **Time Complexity**: O(log₃ n) - Divide into three parts
- **Space Complexity**: O(1) for iterative, O(log₃ n) for recursive
- **When to Use**: Unimodal functions (functions with single peak/valley)
- **Algorithm**: Divide search space into three parts, eliminate one-third
- **Advantages**: Can be faster than binary search for certain functions
- **Disadvantages**: More complex, limited applicability

**4. Exponential Search:**

- **Time Complexity**: O(log n) - Find range then binary search
- **Space Complexity**: O(1)
- **When to Use**: When you don't know the size of the array or when target is near the beginning
- **Algorithm**: Start with small range, double it until target is found, then binary search
- **Advantages**: Works on unbounded arrays, efficient for targets near beginning
- **Disadvantages**: More complex implementation

**Advanced Search Techniques:**

- **Interpolation Search**: O(log log n) average case for uniformly distributed data
- **Jump Search**: O(√n) - Jump by fixed steps, then linear search
- **Fibonacci Search**: O(log n) - Uses Fibonacci numbers to divide search space
- **Hash-based Search**: O(1) average case using hash tables

### **When to Use Binary Search**

**Detailed Explanation:**
Binary search is one of the most powerful and commonly used search algorithms, but it's only applicable under specific conditions. Understanding when and how to use binary search is crucial for solving many algorithmic problems.

**Prerequisites for Binary Search:**

**1. Sorted Array:**

- **Requirement**: The array must be sorted in ascending or descending order
- **Why**: Binary search relies on the ability to eliminate half the search space
- **Implementation**: If array is not sorted, sort it first (O(n log n)) or use linear search
- **Edge Cases**: Handle arrays with duplicate elements carefully

**2. Monotonic Function:**

- **Definition**: A function that is either entirely non-increasing or non-decreasing
- **Examples**: Square root function, power functions, mathematical functions
- **Application**: Use binary search to find the point where function equals target
- **Implementation**: Define search space and comparison function

**3. Search Space:**

- **Definition**: A well-defined range where the answer can exist
- **Examples**: [0, n] for array indices, [0, x] for square root of x
- **Implementation**: Define left and right boundaries clearly
- **Edge Cases**: Handle empty search spaces and single-element spaces

**4. Comparison:**

- **Requirement**: Must be able to compare target with elements in search space
- **Types**: Direct comparison (==, <, >) or function evaluation
- **Implementation**: Define clear comparison logic
- **Edge Cases**: Handle equal elements and boundary conditions

**Binary Search Variants:**

**1. Standard Binary Search:**

- **Purpose**: Find exact target element
- **Return**: Index of target or -1 if not found
- **Implementation**: Use left <= right condition

**2. Lower Bound (First Position):**

- **Purpose**: Find first occurrence of target or insertion point
- **Return**: Leftmost position where target can be inserted
- **Implementation**: Use left < right condition, return left

**3. Upper Bound (Last Position):**

- **Purpose**: Find last occurrence of target or insertion point
- **Return**: Rightmost position where target can be inserted
- **Implementation**: Use left < right condition, return left - 1

**4. Search in Rotated Array:**

- **Purpose**: Search in array that has been rotated
- **Challenge**: Array is not fully sorted
- **Solution**: Identify which half is sorted and apply binary search accordingly

**Common Binary Search Patterns:**

- **Find Peak Element**: Search for local maximum in array
- **Find Minimum in Rotated Array**: Find pivot point in rotated array
- **Square Root**: Find integer square root using binary search
- **Power Function**: Implement fast power using binary search concepts

**Discussion Questions & Answers:**

**Q1: How do you implement binary search efficiently and avoid common pitfalls in Go?**

**Answer:** Efficient binary search implementation:

- **Overflow Prevention**: Use `left + (right-left)/2` instead of `(left+right)/2` to prevent integer overflow
- **Boundary Conditions**: Handle edge cases like empty arrays, single elements, and target not found
- **Loop Condition**: Use `left <= right` for standard search, `left < right` for lower/upper bound
- **Index Updates**: Update `left = mid + 1` and `right = mid - 1` to avoid infinite loops
- **Return Values**: Return appropriate values (-1 for not found, index for found)
- **Type Safety**: Use appropriate data types (int, int64) based on array size
- **Memory Management**: Use iterative approach to avoid stack overflow for large arrays
- **Testing**: Test with edge cases including empty arrays, single elements, and duplicates
- **Performance**: Ensure O(log n) time complexity and O(1) space complexity
- **Documentation**: Document the function behavior and assumptions clearly

**Q2: What are the different variants of binary search and when to use each?**

**Answer:** Binary search variants and usage:

- **Standard Binary Search**: Use when you need to find exact target element in sorted array
- **Lower Bound**: Use when you need to find first occurrence or insertion point for target
- **Upper Bound**: Use when you need to find last occurrence or insertion point for target
- **Search in Rotated Array**: Use when array has been rotated but still partially sorted
- **Search in 2D Matrix**: Use when searching in sorted 2D matrix (treat as 1D array)
- **Find Peak Element**: Use when searching for local maximum in array
- **Find Minimum in Rotated Array**: Use when finding pivot point in rotated array
- **Square Root**: Use when finding integer square root of a number
- **Power Function**: Use when implementing fast power calculation
- **Range Search**: Use when finding range of elements equal to target

**Q3: How do you optimize binary search for different scenarios and handle edge cases?**

**Answer:** Optimization and edge case handling:

- **Array Size**: Use appropriate data types based on array size (int vs int64)
- **Memory Usage**: Prefer iterative implementation over recursive to avoid stack overflow
- **Early Termination**: Return immediately when target is found to avoid unnecessary comparisons
- **Duplicate Handling**: Use lower/upper bound variants when dealing with duplicate elements
- **Empty Arrays**: Handle empty arrays by checking length before starting search
- **Single Element**: Handle single-element arrays as special case
- **Target Not Found**: Return appropriate sentinel value (-1) when target is not found
- **Boundary Values**: Handle cases where target is at beginning or end of array
- **Overflow Prevention**: Use safe midpoint calculation to prevent integer overflow
- **Type Conversions**: Handle type conversions carefully, especially with different integer types
- **Performance Profiling**: Use Go profiling tools to identify performance bottlenecks
- **Testing**: Implement comprehensive tests covering all edge cases and scenarios

---

## 🛠️ Go-Specific Tips

### **Binary Search Template**

```go
func binarySearch(nums []int, target int) int {
    left, right := 0, len(nums)-1

    for left <= right {
        mid := left + (right-left)/2

        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return -1
}
```

### **Lower Bound (First Position)**

```go
func lowerBound(nums []int, target int) int {
    left, right := 0, len(nums)

    for left < right {
        mid := left + (right-left)/2

        if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid
        }
    }

    return left
}
```

### **Upper Bound (Last Position)**

```go
func upperBound(nums []int, target int) int {
    left, right := 0, len(nums)

    for left < right {
        mid := left + (right-left)/2

        if nums[mid] <= target {
            left = mid + 1
        } else {
            right = mid
        }
    }

    return left - 1
}
```

---

## 🎯 Interview Tips

### **How to Identify Search Problems**

1. **Sorted Data**: Array or function is sorted
2. **Find Element**: Need to find specific element
3. **Optimization**: Find minimum/maximum value
4. **Range Query**: Find elements in range

### **Common Search Problem Patterns**

- **Element Search**: Find if element exists
- **Position Search**: Find insertion position
- **Range Search**: Find first/last occurrence
- **Optimization**: Find minimum/maximum

### **Optimization Tips**

- **Use Binary Search**: For sorted data
- **Avoid Overflow**: Use left + (right-left)/2
- **Handle Duplicates**: Use lower/upper bound
- **Early Termination**: Return when found
