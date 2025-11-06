---
# Auto-generated front matter
Title: Findallnumbersdisappearedinarray
LastUpdated: 2025-11-06T20:45:58.720039
Tags: []
Status: draft
---

# Find All Numbers Disappeared in an Array

### Problem
Given an array `nums` of `n` integers where `nums[i]` is in the range `[1, n]`, return an array of all the integers in the range `[1, n]` that do not appear in `nums`.

**Example:**
```
Input: nums = [4,3,2,7,8,2,3,1]
Output: [5,6]

Input: nums = [1,1]
Output: [2]
```

### Golang Solution

```go
func findDisappearedNumbers(nums []int) []int {
    var result []int
    
    // Mark numbers as negative
    for i := 0; i < len(nums); i++ {
        index := abs(nums[i]) - 1
        if nums[index] > 0 {
            nums[index] = -nums[index]
        }
    }
    
    // Find positive numbers (disappeared)
    for i := 0; i < len(nums); i++ {
        if nums[i] > 0 {
            result = append(result, i+1)
        }
    }
    
    return result
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```

### Alternative Solutions

#### **Using Hash Set**
```go
func findDisappearedNumbersHashSet(nums []int) []int {
    numSet := make(map[int]bool)
    var result []int
    
    // Add all numbers to set
    for _, num := range nums {
        numSet[num] = true
    }
    
    // Find missing numbers
    for i := 1; i <= len(nums); i++ {
        if !numSet[i] {
            result = append(result, i)
        }
    }
    
    return result
}
```

#### **Using Array as Hash Map**
```go
func findDisappearedNumbersArray(nums []int) []int {
    n := len(nums)
    present := make([]bool, n+1)
    var result []int
    
    // Mark present numbers
    for _, num := range nums {
        present[num] = true
    }
    
    // Find missing numbers
    for i := 1; i <= n; i++ {
        if !present[i] {
            result = append(result, i)
        }
    }
    
    return result
}
```

#### **Using Sorting**
```go
import "sort"

func findDisappearedNumbersSort(nums []int) []int {
    sort.Ints(nums)
    var result []int
    expected := 1
    
    for i := 0; i < len(nums); i++ {
        for expected < nums[i] {
            result = append(result, expected)
            expected++
        }
        if expected == nums[i] {
            expected++
        }
    }
    
    // Add remaining numbers
    for expected <= len(nums) {
        result = append(result, expected)
        expected++
    }
    
    return result
}
```

#### **Return with Count**
```go
func findDisappearedNumbersWithCount(nums []int) ([]int, int) {
    var result []int
    
    // Mark numbers as negative
    for i := 0; i < len(nums); i++ {
        index := abs(nums[i]) - 1
        if nums[index] > 0 {
            nums[index] = -nums[index]
        }
    }
    
    // Find positive numbers (disappeared)
    for i := 0; i < len(nums); i++ {
        if nums[i] > 0 {
            result = append(result, i+1)
        }
    }
    
    return result, len(result)
}
```

### Complexity
- **Time Complexity:** O(n) for optimal, O(n log n) for sorting
- **Space Complexity:** O(1) for in-place, O(n) for hash set
