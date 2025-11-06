---
# Auto-generated front matter
Title: Threesum
LastUpdated: 2025-11-06T20:45:58.700301
Tags: []
Status: draft
---

# 3Sum

### Problem
Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

**Example:**
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = [0,1,1]
Output: []

Input: nums = [0,0,0]
Output: [[0,0,0]]
```

### Golang Solution

```go
import "sort"

func threeSum(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    
    for i := 0; i < len(nums)-2; i++ {
        // Skip duplicates for the first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        left, right := i+1, len(nums)-1
        target := -nums[i]
        
        for left < right {
            sum := nums[left] + nums[right]
            
            if sum == target {
                result = append(result, []int{nums[i], nums[left], nums[right]})
                
                // Skip duplicates for the second element
                for left < right && nums[left] == nums[left+1] {
                    left++
                }
                // Skip duplicates for the third element
                for left < right && nums[right] == nums[right-1] {
                    right--
                }
                
                left++
                right--
            } else if sum < target {
                left++
            } else {
                right--
            }
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Hash Set**
```go
func threeSumHashSet(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    
    for i := 0; i < len(nums)-2; i++ {
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        seen := make(map[int]bool)
        target := -nums[i]
        
        for j := i + 1; j < len(nums); j++ {
            complement := target - nums[j]
            
            if seen[complement] {
                result = append(result, []int{nums[i], complement, nums[j]})
                
                // Skip duplicates
                for j+1 < len(nums) && nums[j] == nums[j+1] {
                    j++
                }
            }
            
            seen[nums[j]] = true
        }
    }
    
    return result
}
```

#### **Brute Force (Not Recommended)**
```go
func threeSumBruteForce(nums []int) [][]int {
    sort.Ints(nums)
    var result [][]int
    seen := make(map[string]bool)
    
    for i := 0; i < len(nums)-2; i++ {
        for j := i + 1; j < len(nums)-1; j++ {
            for k := j + 1; k < len(nums); k++ {
                if nums[i]+nums[j]+nums[k] == 0 {
                    triplet := []int{nums[i], nums[j], nums[k]}
                    key := fmt.Sprintf("%d,%d,%d", nums[i], nums[j], nums[k])
                    
                    if !seen[key] {
                        result = append(result, triplet)
                        seen[key] = true
                    }
                }
            }
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n²) for two pointers, O(n²) for hash set, O(n³) for brute force
- **Space Complexity:** O(1) for two pointers, O(n) for hash set
