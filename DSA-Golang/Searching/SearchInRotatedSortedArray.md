# Search in Rotated Sorted Array

### Problem
There is an integer array `nums` sorted in ascending order (with distinct values).

Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index `k` (1 <= k < nums.length) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`.

For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` in `nums`, or `-1` if it is not in `nums`.

You must write an algorithm with O(log n) runtime complexity.

**Example:**
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Input: nums = [1], target = 0
Output: -1
```

### Golang Solution

```go
func search(nums []int, target int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            return mid
        }
        
        // Check which half is sorted
        if nums[left] <= nums[mid] {
            // Left half is sorted
            if target >= nums[left] && target < nums[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            // Right half is sorted
            if target > nums[mid] && target <= nums[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    
    return -1
}
```

### Alternative Solutions

#### **Find Pivot First**
```go
func searchFindPivot(nums []int, target int) int {
    pivot := findPivot(nums)
    
    if pivot == -1 {
        return binarySearch(nums, target, 0, len(nums)-1)
    }
    
    if nums[pivot] == target {
        return pivot
    }
    
    if target >= nums[0] {
        return binarySearch(nums, target, 0, pivot-1)
    }
    
    return binarySearch(nums, target, pivot+1, len(nums)-1)
}

func findPivot(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if mid < len(nums)-1 && nums[mid] > nums[mid+1] {
            return mid
        }
        
        if nums[left] <= nums[mid] {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

func binarySearch(nums []int, target, left, right int) int {
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

#### **Recursive Approach**
```go
func searchRecursive(nums []int, target int) int {
    return searchHelper(nums, target, 0, len(nums)-1)
}

func searchHelper(nums []int, target, left, right int) int {
    if left > right {
        return -1
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] == target {
        return mid
    }
    
    if nums[left] <= nums[mid] {
        if target >= nums[left] && target < nums[mid] {
            return searchHelper(nums, target, left, mid-1)
        }
        return searchHelper(nums, target, mid+1, right)
    } else {
        if target > nums[mid] && target <= nums[right] {
            return searchHelper(nums, target, mid+1, right)
        }
        return searchHelper(nums, target, left, mid-1)
    }
}
```

#### **Linear Search (Not Recommended)**
```go
func searchLinear(nums []int, target int) int {
    for i, num := range nums {
        if num == target {
            return i
        }
    }
    return -1
}
```

#### **Find All Occurrences**
```go
func searchAllOccurrences(nums []int, target int) []int {
    var indices []int
    
    for i, num := range nums {
        if num == target {
            indices = append(indices, i)
        }
    }
    
    return indices
}
```

### Complexity
- **Time Complexity:** O(log n) for binary search, O(n) for linear search
- **Space Complexity:** O(1) for iterative, O(log n) for recursive
