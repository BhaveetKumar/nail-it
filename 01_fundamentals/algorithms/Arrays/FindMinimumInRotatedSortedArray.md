---
# Auto-generated front matter
Title: Findminimuminrotatedsortedarray
LastUpdated: 2025-11-06T20:45:58.721910
Tags: []
Status: draft
---

# Find Minimum in Rotated Sorted Array

### Problem
Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

**Example:**
```
Input: nums = [3,4,5,1,2]
Output: 1

Input: nums = [4,5,6,7,0,1,2]
Output: 0

Input: nums = [11,13,15,17]
Output: 11
```

### Golang Solution

```go
func findMin(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[right] {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return nums[left]
}
```

### Alternative Solutions

#### **Linear Search**
```go
func findMinLinear(nums []int) int {
    min := nums[0]
    
    for i := 1; i < len(nums); i++ {
        if nums[i] < min {
            min = nums[i]
        }
    }
    
    return min
}
```

#### **Recursive Binary Search**
```go
func findMinRecursive(nums []int) int {
    return findMinHelper(nums, 0, len(nums)-1)
}

func findMinHelper(nums []int, left, right int) int {
    if left == right {
        return nums[left]
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] > nums[right] {
        return findMinHelper(nums, mid+1, right)
    }
    
    return findMinHelper(nums, left, mid)
}
```

#### **Find Rotation Count**
```go
func findMinWithRotationCount(nums []int) (int, int) {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[right] {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    minElement := nums[left]
    rotationCount := left
    
    return minElement, rotationCount
}
```

#### **Handle Duplicates**
```go
func findMinWithDuplicates(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[right] {
            left = mid + 1
        } else if nums[mid] < nums[right] {
            right = mid
        } else {
            // Handle duplicates
            right--
        }
    }
    
    return nums[left]
}
```

### Complexity
- **Time Complexity:** O(log n) for binary search, O(n) for linear search
- **Space Complexity:** O(1) for iterative, O(log n) for recursive