---
# Auto-generated front matter
Title: Maximumsumsubarrayofsizek
LastUpdated: 2025-11-06T20:45:58.711434
Tags: []
Status: draft
---

# Maximum Sum Subarray of Size K

### Problem
Given an array of positive numbers and a positive number `k`, find the maximum sum of any contiguous subarray of size `k`.

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
    
    windowSum := 0
    maxSum := 0
    
    // Calculate sum of first window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    
    maxSum = windowSum
    
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

#### **Brute Force**
```go
func maxSumSubarrayOfSizeKBruteForce(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    maxSum := 0
    
    for i := 0; i <= len(nums)-k; i++ {
        windowSum := 0
        for j := i; j < i+k; j++ {
            windowSum += nums[j]
        }
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}
```

#### **Return Subarray Indices**
```go
func maxSumSubarrayOfSizeKWithIndices(nums []int, k int) (int, int, int) {
    if len(nums) < k {
        return 0, -1, -1
    }
    
    windowSum := 0
    maxSum := 0
    start := 0
    
    // Calculate sum of first window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    
    maxSum = windowSum
    
    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        if windowSum > maxSum {
            maxSum = windowSum
            start = i - k + 1
        }
    }
    
    return maxSum, start, start + k - 1
}
```

#### **Using Two Pointers**
```go
func maxSumSubarrayOfSizeKTwoPointers(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    left := 0
    windowSum := 0
    maxSum := 0
    
    for right := 0; right < len(nums); right++ {
        windowSum += nums[right]
        
        if right >= k-1 {
            maxSum = max(maxSum, windowSum)
            windowSum -= nums[left]
            left++
        }
    }
    
    return maxSum
}
```

#### **Return All Subarrays of Size K**
```go
func findAllSubarraysOfSizeK(nums []int, k int) [][]int {
    if len(nums) < k {
        return [][]int{}
    }
    
    var result [][]int
    
    for i := 0; i <= len(nums)-k; i++ {
        subarray := make([]int, k)
        copy(subarray, nums[i:i+k])
        result = append(result, subarray)
    }
    
    return result
}
```

#### **Return Subarray with Maximum Sum**
```go
func maxSumSubarrayOfSizeKReturnSubarray(nums []int, k int) []int {
    if len(nums) < k {
        return []int{}
    }
    
    windowSum := 0
    maxSum := 0
    start := 0
    
    // Calculate sum of first window
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }
    
    maxSum = windowSum
    
    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        if windowSum > maxSum {
            maxSum = windowSum
            start = i - k + 1
        }
    }
    
    return nums[start : start+k]
}
```

#### **Using Prefix Sum**
```go
func maxSumSubarrayOfSizeKPrefixSum(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }
    
    prefixSum := make([]int, len(nums)+1)
    
    // Calculate prefix sums
    for i := 0; i < len(nums); i++ {
        prefixSum[i+1] = prefixSum[i] + nums[i]
    }
    
    maxSum := 0
    
    // Find maximum sum of subarray of size k
    for i := 0; i <= len(nums)-k; i++ {
        windowSum := prefixSum[i+k] - prefixSum[i]
        maxSum = max(maxSum, windowSum)
    }
    
    return maxSum
}
```

### Complexity
- **Time Complexity:** O(n) for sliding window, O(n×k) for brute force
- **Space Complexity:** O(1) for sliding window, O(n) for prefix sum