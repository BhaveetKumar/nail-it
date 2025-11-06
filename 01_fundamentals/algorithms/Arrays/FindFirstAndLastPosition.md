---
# Auto-generated front matter
Title: Findfirstandlastposition
LastUpdated: 2025-11-06T20:45:58.724401
Tags: []
Status: draft
---

# Find First and Last Position of Element in Sorted Array

### Problem
Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

**Example:**
```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

### Golang Solution

```go
func searchRange(nums []int, target int) []int {
    left := findFirst(nums, target)
    if left == -1 {
        return []int{-1, -1}
    }
    
    right := findLast(nums, target)
    return []int{left, right}
}

func findFirst(nums []int, target int) int {
    left, right := 0, len(nums)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            result = mid
            right = mid - 1
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}

func findLast(nums []int, target int) int {
    left, right := 0, len(nums)-1
    result := -1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            result = mid
            left = mid + 1
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)
