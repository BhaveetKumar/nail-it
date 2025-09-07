# Search in Rotated Sorted Array

### Problem
There is an integer array `nums` sorted in ascending order (with distinct values).

Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index `k` (1 <= k < nums.length) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (0-indexed).

For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.

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
func searchWithPivot(nums []int, target int) int {
    if len(nums) == 0 {
        return -1
    }
    
    pivot := findPivot(nums)
    
    if target >= nums[0] && target <= nums[pivot] {
        return binarySearch(nums, 0, pivot, target)
    } else {
        return binarySearch(nums, pivot+1, len(nums)-1, target)
    }
}

func findPivot(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[right] {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}

func binarySearch(nums []int, left, right, target int) int {
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

#### **Return All Occurrences**
```go
func searchAll(nums []int, target int) []int {
    var result []int
    
    for i := 0; i < len(nums); i++ {
        if nums[i] == target {
            result = append(result, i)
        }
    }
    
    return result
}
```

#### **Return with Rotation Info**
```go
type SearchResult struct {
    Index    int
    Found    bool
    Pivot    int
    Rotation int
}

func searchWithInfo(nums []int, target int) SearchResult {
    if len(nums) == 0 {
        return SearchResult{Found: false}
    }
    
    pivot := findPivot(nums)
    rotation := pivot
    
    index := -1
    if target >= nums[0] && target <= nums[pivot] {
        index = binarySearch(nums, 0, pivot, target)
    } else {
        index = binarySearch(nums, pivot+1, len(nums)-1, target)
    }
    
    return SearchResult{
        Index:    index,
        Found:    index != -1,
        Pivot:    pivot,
        Rotation: rotation,
    }
}
```

#### **Return Sorted Array**
```go
func restoreSortedArray(nums []int) []int {
    if len(nums) == 0 {
        return []int{}
    }
    
    pivot := findPivot(nums)
    
    // Rotate back to sorted order
    result := make([]int, len(nums))
    copy(result, nums[pivot:])
    copy(result[len(nums)-pivot:], nums[:pivot])
    
    return result
}
```

#### **Return Rotation Count**
```go
func rotationCount(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    pivot := findPivot(nums)
    return pivot
}
```

#### **Check if Array is Rotated**
```go
func isRotated(nums []int) bool {
    if len(nums) <= 1 {
        return false
    }
    
    for i := 1; i < len(nums); i++ {
        if nums[i] < nums[i-1] {
            return true
        }
    }
    
    return false
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)