# Permutations

### Problem
Given an array `nums` of distinct integers, return all the possible permutations. You can return the answer in any order.

**Example:**
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Input: nums = [0,1]
Output: [[0,1],[1,0]]
```

### Golang Solution

```go
func permute(nums []int) [][]int {
    var result [][]int
    var backtrack func([]int)
    
    backtrack = func(current []int) {
        if len(current) == len(nums) {
            permutation := make([]int, len(current))
            copy(permutation, current)
            result = append(result, permutation)
            return
        }
        
        for _, num := range nums {
            if !contains(current, num) {
                current = append(current, num)
                backtrack(current)
                current = current[:len(current)-1] // backtrack
            }
        }
    }
    
    backtrack([]int{})
    return result
}

func contains(slice []int, num int) bool {
    for _, n := range slice {
        if n == num {
            return true
        }
    }
    return false
}
```

### Alternative Solutions

#### **Swap-based Approach**
```go
func permuteSwap(nums []int) [][]int {
    var result [][]int
    permuteHelper(nums, 0, &result)
    return result
}

func permuteHelper(nums []int, start int, result *[][]int) {
    if start == len(nums) {
        permutation := make([]int, len(nums))
        copy(permutation, nums)
        *result = append(*result, permutation)
        return
    }
    
    for i := start; i < len(nums); i++ {
        nums[start], nums[i] = nums[i], nums[start]
        permuteHelper(nums, start+1, result)
        nums[start], nums[i] = nums[i], nums[start] // backtrack
    }
}
```

### Complexity
- **Time Complexity:** O(n! × n)
- **Space Complexity:** O(n! × n)
