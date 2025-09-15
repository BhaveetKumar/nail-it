# Search in Rotated Sorted Array II

### Problem
There is an integer array `nums` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, `nums` is rotated at an unknown pivot index `k` (0 <= k < nums.length) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`.

Given the array `nums` after the possible rotation and an integer `target`, return `true` if `target` is in `nums`, or `false` if it is not in `nums`.

**Example:**
```
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
```

### Golang Solution

```go
func search(nums []int, target int) bool {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            return true
        }
        
        // Handle duplicates
        if nums[left] == nums[mid] && nums[mid] == nums[right] {
            left++
            right--
            continue
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
    
    return false
}
```

### Alternative Solutions

#### **Linear Search (Fallback)**
```go
func searchLinear(nums []int, target int) bool {
    for _, num := range nums {
        if num == target {
            return true
        }
    }
    return false
}
```

### Complexity
- **Time Complexity:** O(log n) average, O(n) worst case
- **Space Complexity:** O(1)
