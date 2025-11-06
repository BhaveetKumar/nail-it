---
# Auto-generated front matter
Title: Partitionequalsubsetsum
LastUpdated: 2025-11-06T20:45:58.746152
Tags: []
Status: draft
---

# Partition Equal Subset Sum

### Problem
Given a non-empty array `nums` containing only positive integers, find if the array can be partitioned into two subsets with equal sum.

**Example:**
```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
```

### Golang Solution

```go
func canPartition(nums []int) bool {
    totalSum := 0
    for _, num := range nums {
        totalSum += num
    }
    
    if totalSum%2 != 0 {
        return false
    }
    
    target := totalSum / 2
    dp := make([]bool, target+1)
    dp[0] = true
    
    for _, num := range nums {
        for j := target; j >= num; j-- {
            dp[j] = dp[j] || dp[j-num]
        }
    }
    
    return dp[target]
}
```

### Alternative Solutions

#### **2D DP Approach**
```go
func canPartition2D(nums []int) bool {
    totalSum := 0
    for _, num := range nums {
        totalSum += num
    }
    
    if totalSum%2 != 0 {
        return false
    }
    
    target := totalSum / 2
    n := len(nums)
    dp := make([][]bool, n+1)
    for i := range dp {
        dp[i] = make([]bool, target+1)
        dp[i][0] = true
    }
    
    for i := 1; i <= n; i++ {
        for j := 1; j <= target; j++ {
            if j < nums[i-1] {
                dp[i][j] = dp[i-1][j]
            } else {
                dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i-1]]
            }
        }
    }
    
    return dp[n][target]
}
```

#### **DFS with Memoization**
```go
func canPartitionDFS(nums []int) bool {
    totalSum := 0
    for _, num := range nums {
        totalSum += num
    }
    
    if totalSum%2 != 0 {
        return false
    }
    
    target := totalSum / 2
    memo := make(map[string]bool)
    
    var dfs func(int, int) bool
    dfs = func(index, sum int) bool {
        if sum == target {
            return true
        }
        if sum > target || index >= len(nums) {
            return false
        }
        
        key := fmt.Sprintf("%d,%d", index, sum)
        if val, exists := memo[key]; exists {
            return val
        }
        
        result := dfs(index+1, sum) || dfs(index+1, sum+nums[index])
        memo[key] = result
        return result
    }
    
    return dfs(0, 0)
}
```

### Complexity
- **Time Complexity:** O(n × sum) for DP, O(2^n) for DFS
- **Space Complexity:** O(sum) for 1D DP, O(n × sum) for 2D DP
