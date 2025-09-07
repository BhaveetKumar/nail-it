# First Missing Positive

### Problem
Given an unsorted integer array `nums`, return the smallest missing positive integer.

You must implement an algorithm that runs in `O(n)` time and uses constant extra space.

**Example:**
```
Input: nums = [1,2,0]
Output: 3

Input: nums = [3,4,-1,1]
Output: 2

Input: nums = [7,8,9,11,12]
Output: 1
```

### Golang Solution

```go
func firstMissingPositive(nums []int) int {
    n := len(nums)
    
    // Step 1: Replace negative numbers, zeros, and numbers > n with n+1
    for i := 0; i < n; i++ {
        if nums[i] <= 0 || nums[i] > n {
            nums[i] = n + 1
        }
    }
    
    // Step 2: Use array indices as hash keys
    for i := 0; i < n; i++ {
        num := abs(nums[i])
        if num <= n {
            nums[num-1] = -abs(nums[num-1])
        }
    }
    
    // Step 3: Find first positive number
    for i := 0; i < n; i++ {
        if nums[i] > 0 {
            return i + 1
        }
    }
    
    return n + 1
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
