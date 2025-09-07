# Subarray Sum Equals K

### Problem
Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

A subarray is a contiguous non-empty sequence of elements within an array.

**Example:**
```
Input: nums = [1,1,1], k = 2
Output: 2

Input: nums = [1,2,3], k = 3
Output: 2
```

### Golang Solution

```go
func subarraySum(nums []int, k int) int {
    count := 0
    sum := 0
    sumCount := make(map[int]int)
    sumCount[0] = 1 // Empty subarray has sum 0
    
    for _, num := range nums {
        sum += num
        
        // If (sum - k) exists in map, then there are sumCount[sum-k] subarrays
        // ending at current position with sum k
        if freq, exists := sumCount[sum-k]; exists {
            count += freq
        }
        
        // Update the count of current sum
        sumCount[sum]++
    }
    
    return count
}
```

### Alternative Solutions

#### **Brute Force**
```go
func subarraySumBruteForce(nums []int, k int) int {
    count := 0
    
    for i := 0; i < len(nums); i++ {
        sum := 0
        for j := i; j < len(nums); j++ {
            sum += nums[j]
            if sum == k {
                count++
            }
        }
    }
    
    return count
}
```

#### **Prefix Sum with Array**
```go
func subarraySumPrefixArray(nums []int, k int) int {
    n := len(nums)
    prefixSum := make([]int, n+1)
    
    // Calculate prefix sums
    for i := 0; i < n; i++ {
        prefixSum[i+1] = prefixSum[i] + nums[i]
    }
    
    count := 0
    
    for i := 0; i < n; i++ {
        for j := i + 1; j <= n; j++ {
            if prefixSum[j]-prefixSum[i] == k {
                count++
            }
        }
    }
    
    return count
}
```

### Complexity
- **Time Complexity:** O(n) for hash map, O(n²) for brute force
- **Space Complexity:** O(n)