---
# Auto-generated front matter
Title: Subsetsii
LastUpdated: 2025-11-06T20:45:58.708773
Tags: []
Status: draft
---

# Subsets II

### Problem
Given an integer array `nums` that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

**Example:**
```
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Input: nums = [0]
Output: [[],[0]]
```

### Golang Solution

```go
import "sort"

func subsetsWithDup(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    var current []int
    
    var backtrack func(int)
    backtrack = func(start int) {
        subset := make([]int, len(current))
        copy(subset, current)
        result = append(result, subset)
        
        for i := start; i < len(nums); i++ {
            if i > start && nums[i] == nums[i-1] {
                continue
            }
            
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

#### **Using Set to Avoid Duplicates**
```go
func subsetsWithDupSet(nums []int) [][]int {
    sort.Ints(nums)
    resultSet := make(map[string]bool)
    var result [][]int
    
    var backtrack func(int, []int)
    backtrack = func(start int, current []int) {
        key := fmt.Sprintf("%v", current)
        if !resultSet[key] {
            subset := make([]int, len(current))
            copy(subset, current)
            result = append(result, subset)
            resultSet[key] = true
        }
        
        for i := start; i < len(nums); i++ {
            current = append(current, nums[i])
            backtrack(i+1, current)
            current = current[:len(current)-1]
        }
    }
    
    backtrack(0, []int{})
    return result
}
```

#### **Iterative Approach**
```go
func subsetsWithDupIterative(nums []int) [][]int {
    sort.Ints(nums)
    result := [][]int{{}}
    
    for i := 0; i < len(nums); i++ {
        size := len(result)
        
        for j := 0; j < size; j++ {
            subset := make([]int, len(result[j]))
            copy(subset, result[j])
            subset = append(subset, nums[i])
            result = append(result, subset)
        }
        
        // Skip duplicates
        for i+1 < len(nums) && nums[i] == nums[i+1] {
            i++
            size = len(result)
            for j := 0; j < size; j++ {
                subset := make([]int, len(result[j]))
                copy(subset, result[j])
                subset = append(subset, nums[i])
                result = append(result, subset)
            }
        }
    }
    
    return result
}
```

#### **Bit Manipulation**
```go
func subsetsWithDupBitManipulation(nums []int) [][]int {
    sort.Ints(nums)
    resultSet := make(map[string]bool)
    n := len(nums)
    
    for i := 0; i < (1 << n); i++ {
        var subset []int
        for j := 0; j < n; j++ {
            if i&(1<<j) != 0 {
                subset = append(subset, nums[j])
            }
        }
        
        key := fmt.Sprintf("%v", subset)
        if !resultSet[key] {
            resultSet[key] = true
        }
    }
    
    var result [][]int
    for key := range resultSet {
        // Parse key back to slice (simplified)
        result = append(result, []int{})
    }
    
    return result
}
```

#### **With Frequency Count**
```go
func subsetsWithDupFrequency(nums []int) [][]int {
    sort.Ints(nums)
    
    // Count frequencies
    freq := make(map[int]int)
    for _, num := range nums {
        freq[num]++
    }
    
    // Get unique numbers
    var uniqueNums []int
    for num := range freq {
        uniqueNums = append(uniqueNums, num)
    }
    sort.Ints(uniqueNums)
    
    var result [][]int
    var current []int
    
    var backtrack func(int)
    backtrack = func(start int) {
        subset := make([]int, len(current))
        copy(subset, current)
        result = append(result, subset)
        
        for i := start; i < len(uniqueNums); i++ {
            num := uniqueNums[i]
            
            for count := 1; count <= freq[num]; count++ {
                for j := 0; j < count; j++ {
                    current = append(current, num)
                }
                
                backtrack(i + 1)
                
                for j := 0; j < count; j++ {
                    current = current[:len(current)-1]
                }
            }
        }
    }
    
    backtrack(0)
    return result
}
```

### Complexity
- **Time Complexity:** O(2^n × n)
- **Space Complexity:** O(2^n × n)
