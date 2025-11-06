---
# Auto-generated front matter
Title: Searchinsertposition
LastUpdated: 2025-11-06T20:45:58.706059
Tags: []
Status: draft
---

# Search Insert Position

### Problem
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

**Example:**
```
Input: nums = [1,3,5,6], target = 5
Output: 2

Input: nums = [1,3,5,6], target = 2
Output: 1

Input: nums = [1,3,5,6], target = 7
Output: 4
```

### Golang Solution

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

### Alternative Solutions

#### **Recursive Binary Search**
```go
func searchInsertRecursive(nums []int, target int) int {
    return searchInsertHelper(nums, target, 0, len(nums)-1)
}

func searchInsertHelper(nums []int, target, left, right int) int {
    if left > right {
        return left
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] == target {
        return mid
    } else if nums[mid] < target {
        return searchInsertHelper(nums, target, mid+1, right)
    } else {
        return searchInsertHelper(nums, target, left, mid-1)
    }
}
```

#### **Linear Search**
```go
func searchInsertLinear(nums []int, target int) int {
    for i, num := range nums {
        if num >= target {
            return i
        }
    }
    return len(nums)
}
```

#### **Using Built-in Search**
```go
import "sort"

func searchInsertBuiltin(nums []int, target int) int {
    return sort.SearchInts(nums, target)
}
```

#### **Find All Insert Positions**
```go
func searchInsertAll(nums []int, target int) []int {
    var positions []int
    
    for i, num := range nums {
        if num >= target {
            positions = append(positions, i)
        }
    }
    
    positions = append(positions, len(nums))
    return positions
}
```

#### **With Duplicates**
```go
func searchInsertWithDuplicates(nums []int, target int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return left
}
```

### Complexity
- **Time Complexity:** O(log n) for binary search, O(n) for linear search
- **Space Complexity:** O(1) for iterative, O(log n) for recursive