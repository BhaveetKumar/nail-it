---
# Auto-generated front matter
Title: Permutations
LastUpdated: 2025-11-06T20:45:58.710079
Tags: []
Status: draft
---

# Permutations

### Problem
Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in any order.

**Example:**
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Input: nums = [0,1]
Output: [[0,1],[1,0]]

Input: nums = [1]
Output: [[1]]
```

### Golang Solution

```go
func permute(nums []int) [][]int {
    var result [][]int
    var current []int
    used := make([]bool, len(nums))
    
    var backtrack func()
    backtrack = func() {
        if len(current) == len(nums) {
            permutation := make([]int, len(current))
            copy(permutation, current)
            result = append(result, permutation)
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if !used[i] {
                used[i] = true
                current = append(current, nums[i])
                backtrack()
                current = current[:len(current)-1]
                used[i] = false
            }
        }
    }
    
    backtrack()
    return result
}
```

### Alternative Solutions

#### **Using Swap**
```go
func permuteSwap(nums []int) [][]int {
    var result [][]int
    
    var backtrack func(int)
    backtrack = func(start int) {
        if start == len(nums) {
            permutation := make([]int, len(nums))
            copy(permutation, nums)
            result = append(result, permutation)
            return
        }
        
        for i := start; i < len(nums); i++ {
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
        }
    }
    
    backtrack(0)
    return result
}
```

#### **Using Heap's Algorithm**
```go
func permuteHeap(nums []int) [][]int {
    var result [][]int
    
    var generate func(int)
    generate = func(k int) {
        if k == 1 {
            permutation := make([]int, len(nums))
            copy(permutation, nums)
            result = append(result, permutation)
            return
        }
        
        generate(k - 1)
        
        for i := 0; i < k-1; i++ {
            if k%2 == 0 {
                nums[i], nums[k-1] = nums[k-1], nums[i]
            } else {
                nums[0], nums[k-1] = nums[k-1], nums[0]
            }
            generate(k - 1)
        }
    }
    
    generate(len(nums))
    return result
}
```

#### **Return Permutations with Count**
```go
func permuteWithCount(nums []int) ([][]int, int) {
    var result [][]int
    var current []int
    used := make([]bool, len(nums))
    
    var backtrack func()
    backtrack = func() {
        if len(current) == len(nums) {
            permutation := make([]int, len(current))
            copy(permutation, current)
            result = append(result, permutation)
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if !used[i] {
                used[i] = true
                current = append(current, nums[i])
                backtrack()
                current = current[:len(current)-1]
                used[i] = false
            }
        }
    }
    
    backtrack()
    return result, len(result)
}
```

#### **Return Next Permutation**
```go
func nextPermutation(nums []int) {
    i := len(nums) - 2
    
    // Find the largest index i such that nums[i] < nums[i+1]
    for i >= 0 && nums[i] >= nums[i+1] {
        i--
    }
    
    if i >= 0 {
        j := len(nums) - 1
        
        // Find the largest index j such that nums[i] < nums[j]
        for j >= 0 && nums[j] <= nums[i] {
            j--
        }
        
        // Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]
    }
    
    // Reverse the suffix starting at nums[i+1]
    reverse(nums, i+1)
}

func reverse(nums []int, start int) {
    i, j := start, len(nums)-1
    for i < j {
        nums[i], nums[j] = nums[j], nums[i]
        i++
        j--
    }
}
```

#### **Return All Permutations of Length K**
```go
func permuteK(nums []int, k int) [][]int {
    var result [][]int
    var current []int
    used := make([]bool, len(nums))
    
    var backtrack func()
    backtrack = func() {
        if len(current) == k {
            permutation := make([]int, len(current))
            copy(permutation, current)
            result = append(result, permutation)
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if !used[i] {
                used[i] = true
                current = append(current, nums[i])
                backtrack()
                current = current[:len(current)-1]
                used[i] = false
            }
        }
    }
    
    backtrack()
    return result
}
```

#### **Return Permutations with Duplicates**
```go
import "sort"

func permuteUnique(nums []int) [][]int {
    var result [][]int
    var current []int
    used := make([]bool, len(nums))
    
    // Sort to handle duplicates
    sort.Ints(nums)
    
    var backtrack func()
    backtrack = func() {
        if len(current) == len(nums) {
            permutation := make([]int, len(current))
            copy(permutation, current)
            result = append(result, permutation)
            return
        }
        
        for i := 0; i < len(nums); i++ {
            if used[i] || (i > 0 && nums[i] == nums[i-1] && !used[i-1]) {
                continue
            }
            
            used[i] = true
            current = append(current, nums[i])
            backtrack()
            current = current[:len(current)-1]
            used[i] = false
        }
    }
    
    backtrack()
    return result
}
```

### Complexity
- **Time Complexity:** O(n! × n) where n is the length of the array
- **Space Complexity:** O(n! × n)