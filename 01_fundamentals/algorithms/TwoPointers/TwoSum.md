# Two Sum (Two Pointers)

### Problem
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

**Example:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Input: nums = [3,2,4], target = 6
Output: [1,2]
```

### Golang Solution

```go
func twoSum(nums []int, target int) []int {
    // Create a map to store value -> index
    numMap := make(map[int]int)
    
    for i, num := range nums {
        complement := target - num
        if index, exists := numMap[complement]; exists {
            return []int{index, i}
        }
        numMap[num] = i
    }
    
    return nil
}
```

### Alternative Solutions

#### **Two Pointers (for sorted array)**
```go
func twoSumSorted(nums []int, target int) []int {
    left, right := 0, len(nums)-1
    
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    
    return nil
}
```

### Complexity
- **Time Complexity:** O(n) for hash map, O(n log n) for two pointers (due to sorting)
- **Space Complexity:** O(n) for hash map, O(1) for two pointers
