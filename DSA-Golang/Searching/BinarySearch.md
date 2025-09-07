# Binary Search

### Problem
Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with O(log n) runtime complexity.

**Example:**
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

### Golang Solution

```go
func search(nums []int, target int) int {
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

### Alternative Solutions

#### **Recursive Binary Search**
```go
func searchRecursive(nums []int, target int) int {
    return binarySearchHelper(nums, target, 0, len(nums)-1)
}

func binarySearchHelper(nums []int, target, left, right int) int {
    if left > right {
        return -1
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] == target {
        return mid
    } else if nums[mid] < target {
        return binarySearchHelper(nums, target, mid+1, right)
    } else {
        return binarySearchHelper(nums, target, left, mid-1)
    }
}
```

#### **Find First Occurrence**
```go
func findFirstOccurrence(nums []int, target int) int {
    left, right := 0, len(nums)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            result = mid
            right = mid - 1 // Continue searching in left half
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}
```

#### **Find Last Occurrence**
```go
func findLastOccurrence(nums []int, target int) int {
    left, right := 0, len(nums)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            result = mid
            left = mid + 1 // Continue searching in right half
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}
```

#### **Find Insert Position**
```go
func searchInsert(nums []int, target int) int {
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
    
    return left
}
```

#### **Binary Search in 2D Matrix**
```go
func searchMatrix(matrix [][]int, target int) bool {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return false
    }
    
    rows, cols := len(matrix), len(matrix[0])
    left, right := 0, rows*cols-1
    
    for left <= right {
        mid := left + (right-left)/2
        midValue := matrix[mid/cols][mid%cols]
        
        if midValue == target {
            return true
        } else if midValue < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return false
}
```

#### **Return All Occurrences**
```go
func findAllOccurrences(nums []int, target int) []int {
    first := findFirstOccurrence(nums, target)
    if first == -1 {
        return []int{}
    }
    
    last := findLastOccurrence(nums, target)
    var result []int
    
    for i := first; i <= last; i++ {
        result = append(result, i)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1) for iterative, O(log n) for recursive