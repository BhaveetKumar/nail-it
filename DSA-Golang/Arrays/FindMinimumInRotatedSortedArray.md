# Find Minimum in Rotated Sorted Array

### Problem
Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

- `[4,5,6,7,0,1,2]` if it was rotated `4` times.
- `[0,1,2,4,5,6,7]` if it was rotated `7` times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of unique elements, return the minimum element of this array.

**Example:**
```
Input: nums = [3,4,5,1,2]
Output: 1

Input: nums = [4,5,6,7,0,1,2]
Output: 0
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
            right--
        }
    }
    
    return nums[left]
}
```

### Complexity
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)
