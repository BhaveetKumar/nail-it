# 4Sum

### Problem
Given an array `nums` of `n` integers, return an array of all the unique quadruplets `[nums[a], nums[b], nums[c], nums[d]]` such that:

- `0 <= a, b, c, d < n`
- `a`, `b`, `c`, and `d` are distinct.
- `nums[a] + nums[b] + nums[c] + nums[d] == target`

You may return the answer in any order.

**Example:**
```
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]
```

### Golang Solution

```go
import "sort"

func fourSum(nums []int, target int) [][]int {
    sort.Ints(nums)
    var result [][]int
    
    for i := 0; i < len(nums)-3; i++ {
        // Skip duplicates for first element
        if i > 0 && nums[i] == nums[i-1] {
            continue
        }
        
        for j := i + 1; j < len(nums)-2; j++ {
            // Skip duplicates for second element
            if j > i+1 && nums[j] == nums[j-1] {
                continue
            }
            
            left, right := j+1, len(nums)-1
            targetSum := target - nums[i] - nums[j]
            
            for left < right {
                sum := nums[left] + nums[right]
                
                if sum == targetSum {
                    result = append(result, []int{nums[i], nums[j], nums[left], nums[right]})
                    
                    // Skip duplicates for third element
                    for left < right && nums[left] == nums[left+1] {
                        left++
                    }
                    // Skip duplicates for fourth element
                    for left < right && nums[right] == nums[right-1] {
                        right--
                    }
                    
                    left++
                    right--
                } else if sum < targetSum {
                    left++
                } else {
                    right--
                }
            }
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Hash Set**
```go
func fourSumHashSet(nums []int, target int) [][]int {
    sort.Ints(nums)
    var result [][]int
    seen := make(map[string]bool)
    
    for i := 0; i < len(nums)-3; i++ {
        for j := i + 1; j < len(nums)-2; j++ {
            for k := j + 1; k < len(nums)-1; k++ {
                for l := k + 1; l < len(nums); l++ {
                    if nums[i]+nums[j]+nums[k]+nums[l] == target {
                        key := fmt.Sprintf("%d,%d,%d,%d", nums[i], nums[j], nums[k], nums[l])
                        if !seen[key] {
                            result = append(result, []int{nums[i], nums[j], nums[k], nums[l]})
                            seen[key] = true
                        }
                    }
                }
            }
        }
    }
    
    return result
}
```

#### **Generalized K-Sum**
```go
func fourSumGeneralized(nums []int, target int) [][]int {
    sort.Ints(nums)
    return kSum(nums, target, 0, 4)
}

func kSum(nums []int, target int, start int, k int) [][]int {
    var result [][]int
    
    if start == len(nums) || nums[start]*k > target || target > nums[len(nums)-1]*k {
        return result
    }
    
    if k == 2 {
        return twoSum(nums, target, start)
    }
    
    for i := start; i < len(nums); i++ {
        if i == start || nums[i-1] != nums[i] {
            for _, subset := range kSum(nums, target-nums[i], i+1, k-1) {
                result = append(result, append([]int{nums[i]}, subset...))
            }
        }
    }
    
    return result
}

func twoSum(nums []int, target int, start int) [][]int {
    var result [][]int
    left, right := start, len(nums)-1
    
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            result = append(result, []int{nums[left], nums[right]})
            for left < right && nums[left] == nums[left+1] {
                left++
            }
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
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n³) for two pointers, O(n⁴) for hash set
- **Space Complexity:** O(1) for two pointers, O(n) for hash set
