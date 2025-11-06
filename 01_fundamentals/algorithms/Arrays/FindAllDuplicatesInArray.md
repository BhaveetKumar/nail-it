---
# Auto-generated front matter
Title: Findallduplicatesinarray
LastUpdated: 2025-11-06T20:45:58.726585
Tags: []
Status: draft
---

# Find All Duplicates in an Array

### Problem
Given an integer array `nums` of length `n` where all the integers of `nums` are in the range `[1, n]` and each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.

**Example:**
```
Input: nums = [4,3,2,7,8,2,3,1]
Output: [2,3]

Input: nums = [1,1,2]
Output: [1]

Input: nums = [1]
Output: []
```

### Golang Solution

```go
func findDuplicates(nums []int) []int {
    var result []int
    
    for i := 0; i < len(nums); i++ {
        index := abs(nums[i]) - 1
        
        if nums[index] < 0 {
            result = append(result, abs(nums[i]))
        } else {
            nums[index] = -nums[index]
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
func findDuplicatesHashSet(nums []int) []int {
    seen := make(map[int]bool)
    var result []int
    
    for _, num := range nums {
        if seen[num] {
            result = append(result, num)
        } else {
            seen[num] = true
        }
    }
    
    return result
}
```

#### **Using Array as Hash Map**
```go
func findDuplicatesArray(nums []int) []int {
    n := len(nums)
    count := make([]int, n+1)
    var result []int
    
    for _, num := range nums {
        count[num]++
        if count[num] == 2 {
            result = append(result, num)
        }
    }
    
    return result
}
```

#### **Using Sorting**
```go
import "sort"

func findDuplicatesSort(nums []int) []int {
    sort.Ints(nums)
    var result []int
    
    for i := 1; i < len(nums); i++ {
        if nums[i] == nums[i-1] {
            result = append(result, nums[i])
        }
    }
    
    return result
}
```

#### **Using XOR (Limited Use Case)**
```go
func findDuplicatesXOR(nums []int) []int {
    var result []int
    n := len(nums)
    
    for i := 0; i < n; i++ {
        for j := i + 1; j < n; j++ {
            if nums[i] == nums[j] {
                result = append(result, nums[i])
            }
        }
    }
    
    return result
}
```

#### **Return All Occurrences**
```go
func findAllDuplicates(nums []int) []int {
    var result []int
    
    for i := 0; i < len(nums); i++ {
        index := abs(nums[i]) - 1
        
        if nums[index] < 0 {
            result = append(result, abs(nums[i]))
        } else {
            nums[index] = -nums[index]
        }
    }
    
    // Restore original array
    for i := 0; i < len(nums); i++ {
        nums[i] = abs(nums[i])
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for optimal, O(n log n) for sorting
- **Space Complexity:** O(1) for in-place, O(n) for hash set
