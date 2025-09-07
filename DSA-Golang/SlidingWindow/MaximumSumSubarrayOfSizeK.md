# Maximum Sum Subarray of Size K

### Problem
Given an array of positive numbers and a positive number 'k', find the maximum sum of any contiguous subarray of size k.

**Example:**
```
Input: [2, 1, 5, 1, 3, 2], k=3
Output: 9
Explanation: Subarray with maximum sum is [5, 1, 3].

Input: [2, 3, 4, 1, 5], k=2
Output: 7
Explanation: Subarray with maximum sum is [3, 4].
```

### Golang Solution

```go
func maxSumSubarrayOfSizeK(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    // Calculate sum of first window
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    
    maxSum := windowSum
    
    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Two Pointers Approach**
```go
func maxSumSubarrayTwoPointers(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    left, right := 0, 0
    windowSum := 0
    maxSum := 0
    
    for right < len(nums) {
        windowSum += nums[right]
        
        if right-left+1 == k {
            maxSum = max(maxSum, windowSum)
            windowSum -= nums[left]
            left++
        }
        
        right++
    }
    
    return maxSum
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
