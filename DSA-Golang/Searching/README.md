# Searching Pattern

> **Master searching algorithms and their applications with Go implementations**

## 📋 Problems

### **Binary Search**
- [Binary Search](./BinarySearch.md) - Search in sorted array
- [Search Insert Position](./SearchInsertPosition.md) - Find insertion point
- [Find First and Last Position](./FindFirstAndLastPosition.md) - Search range
- [Search in Rotated Sorted Array](./SearchInRotatedSortedArray.md) - Search in rotated array
- [Find Minimum in Rotated Sorted Array](./FindMinimumInRotatedSortedArray.md) - Find minimum

### **Advanced Searching**
- [Search a 2D Matrix](./Search2DMatrix.md) - Search in 2D matrix
- [Search in Rotated Sorted Array II](./SearchInRotatedSortedArrayII.md) - With duplicates
- [Find Peak Element](./FindPeakElement.md) - Find local maximum
- [Sqrt(x)](./Sqrt.md) - Binary search for square root
- [Pow(x, n)](./Pow.md) - Fast power calculation

### **Search Applications**
- [Guess Number Higher or Lower](./GuessNumber.md) - Interactive search
- [First Bad Version](./FirstBadVersion.md) - Find first occurrence
- [Kth Smallest Element](./KthSmallestElement.md) - Find kth element
- [Median of Two Sorted Arrays](./MedianOfTwoSortedArrays.md) - Find median
- [Search Suggestions System](./SearchSuggestionsSystem.md) - Prefix search

---

## 🎯 Key Concepts

### **Search Algorithms**
- **Linear Search**: O(n) - Check each element
- **Binary Search**: O(log n) - Divide and conquer
- **Ternary Search**: O(log₃ n) - Divide into three parts
- **Exponential Search**: O(log n) - Find range then binary search

### **When to Use Binary Search**
- **Sorted Array**: Array is sorted
- **Monotonic Function**: Function is monotonic
- **Search Space**: Can define search space
- **Comparison**: Can compare with target

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
