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
    sumCount[0] = 1
    
    for _, num := range nums {
        sum += num
        if freq, exists := sumCount[sum-k]; exists {
            count += freq
        }
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

#### **Using Prefix Sum Array**
```go
func subarraySumPrefixArray(nums []int, k int) int {
    n := len(nums)
    prefixSum := make([]int, n+1)
    
    // Calculate prefix sums
    for i := 0; i < n; i++ {
        prefixSum[i+1] = prefixSum[i] + nums[i]
    }
    
    count := 0
    
    // Check all subarrays
    for i := 0; i < n; i++ {
        for j := i; j < n; j++ {
            if prefixSum[j+1] - prefixSum[i] == k {
                count++
            }
        }
    }
    
    return count
}
```

#### **Return Subarray Indices**
```go
func subarraySumWithIndices(nums []int, k int) (int, [][]int) {
    count := 0
    sum := 0
    sumIndices := make(map[int][]int)
    sumIndices[0] = []int{-1}
    var subarrays [][]int
    
    for i, num := range nums {
        sum += num
        
        if indices, exists := sumIndices[sum-k]; exists {
            count += len(indices)
            for _, start := range indices {
                subarrays = append(subarrays, []int{start + 1, i})
            }
        }
        
        sumIndices[sum] = append(sumIndices[sum], i)
    }
    
    return count, subarrays
}
```

#### **Using Sliding Window (Only for Positive Numbers)**
```go
func subarraySumSlidingWindow(nums []int, k int) int {
    count := 0
    left := 0
    sum := 0
    
    for right := 0; right < len(nums); right++ {
        sum += nums[right]
        
        for sum > k && left <= right {
            sum -= nums[left]
            left++
        }
        
        if sum == k {
            count++
        }
    }
    
    return count
}
```

#### **Return All Subarrays**
```go
func findAllSubarraysWithSum(nums []int, k int) [][]int {
    var result [][]int
    sum := 0
    sumIndices := make(map[int][]int)
    sumIndices[0] = []int{-1}
    
    for i, num := range nums {
        sum += num
        
        if indices, exists := sumIndices[sum-k]; exists {
            for _, start := range indices {
                subarray := make([]int, i-start)
                copy(subarray, nums[start+1:i+1])
                result = append(result, subarray)
            }
        }
        
        sumIndices[sum] = append(sumIndices[sum], i)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for optimal, O(n²) for brute force
- **Space Complexity:** O(n) for hash map