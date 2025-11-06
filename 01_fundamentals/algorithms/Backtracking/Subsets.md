---
# Auto-generated front matter
Title: Subsets
LastUpdated: 2025-11-06T20:45:58.707381
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
    var current []int
    
    var backtrack func(int)
    backtrack = func(start int) {
        subset := make([]int, len(current))
        copy(subset, current)
        result = append(result, subset)
        
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i + 1)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0)
    return result
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
func subsetsIterative(nums []int) [][]int {
    result := [][]int{{}}
    
    for _, num := range nums {
        size := len(result)
        for i := 0; i < size; i++ {
            subset := make([]int, len(result[i]))
            copy(subset, result[i])
            subset = append(subset, num)
            result = append(result, subset)
        }
    }
    
    return result
}
```

#### **Using Bit Manipulation**
```go
func subsetsBitManipulation(nums []int) [][]int {
    n := len(nums)
    var result [][]int
    
    for i := 0; i < (1 << n); i++ {
        var subset []int
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

#### **Using Recursion**
```go
func subsetsRecursive(nums []int) [][]int {
    if len(nums) == 0 {
        return [][]int{{}}
    }
    
    first := nums[0]
    rest := nums[1:]
    
    subsetsWithoutFirst := subsetsRecursive(rest)
    var result [][]int
    
    // Add subsets without first element
    for _, subset := range subsetsWithoutFirst {
        result = append(result, subset)
    }
    
    // Add subsets with first element
    for _, subset := range subsetsWithoutFirst {
        newSubset := make([]int, len(subset))
        copy(newSubset, subset)
        newSubset = append(newSubset, first)
        result = append(result, newSubset)
    }
    
    return result
}
```

#### **Return Subsets with Size**
```go
func subsetsWithSize(nums []int, k int) [][]int {
    var result [][]int
    var current []int
    
    var backtrack func(int)
    backtrack = func(start int) {
        if len(current) == k {
            subset := make([]int, len(current))
            copy(subset, current)
            result = append(result, subset)
            return
        }
        
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i + 1)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0)
    return result
}
```

#### **Return All Subsets with Counts**
```go
func subsetsWithCounts(nums []int) ([][]int, map[int]int) {
    var result [][]int
    var current []int
    sizeCount := make(map[int]int)
    
    var backtrack func(int)
    backtrack = func(start int) {
        subset := make([]int, len(current))
        copy(subset, current)
        result = append(result, subset)
        sizeCount[len(subset)]++
        
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i + 1)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0)
    return result, sizeCount
}
```

### Complexity
- **Time Complexity:** O(2^n × n) where n is the length of nums
- **Space Complexity:** O(2^n × n)