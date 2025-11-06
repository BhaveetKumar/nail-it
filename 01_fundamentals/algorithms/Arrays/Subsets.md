---
# Auto-generated front matter
Title: Subsets
LastUpdated: 2025-11-06T20:45:58.719514
Tags: []
Status: draft
---

# Subsets

### Problem
Given an integer array `nums` of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

**Example:**
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Input: nums = [0]
Output: [[],[0]]
```

### Golang Solution

```go
func subsets(nums []int) [][]int {
    var result [][]int
    var backtrack func(int, []int)
    
    backtrack = func(start int, current []int) {
        // Add current subset to result
        subset := make([]int, len(current))
        copy(subset, current)
        result = append(result, subset)
        
        // Try adding each remaining element
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i+1, current)
            current = current[:len(current)-1] // backtrack
        }
    }
    
    backtrack(0, []int{})
    return result
}
```

### Alternative Solutions

#### **Bit Manipulation**
```go
func subsetsBitManipulation(nums []int) [][]int {
    n := len(nums)
    totalSubsets := 1 << n
    result := make([][]int, 0, totalSubsets)
    
    for i := 0; i < totalSubsets; i++ {
        subset := []int{}
        for j := 0; j < n; j++ {
            if i&(1<<j) != 0 {
                subset = append(subset, nums[j])
            }
        }
        result = append(result, subset)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(2^n × n)
- **Space Complexity:** O(2^n × n)
