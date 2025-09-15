# Find the Duplicate Number

### Problem
Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only one repeated number in `nums`, return this repeated number.

You must solve the problem without modifying the array `nums` and uses only constant extra space.

**Example:**
```
Input: nums = [1,3,4,2,2]
Output: 2

Input: nums = [3,1,3,4,2]
Output: 3
```

### Golang Solution

```go
func findDuplicate(nums []int) int {
    // Floyd's cycle detection algorithm
    slow := nums[0]
    fast := nums[0]
    
    // Phase 1: Find intersection point
    for {
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast {
            break
        }
    }
    
    // Phase 2: Find entrance to cycle
    slow = nums[0]
    for slow != fast {
        slow = nums[slow]
        fast = nums[fast]
    }
    
    return slow
}
```

### Alternative Solutions

#### **Using Binary Search**
```go
func findDuplicateBinarySearch(nums []int) int {
    left, right := 1, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        count := 0
        
        // Count numbers <= mid
        for _, num := range nums {
            if num <= mid {
                count++
            }
        }
        
        if count > mid {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return left
}
```

#### **Using Hash Set**
```go
func findDuplicateHashSet(nums []int) int {
    seen := make(map[int]bool)
    
    for _, num := range nums {
        if seen[num] {
            return num
        }
        seen[num] = true
    }
    
    return -1
}
```

#### **Using Array as Hash Map**
```go
func findDuplicateArray(nums []int) int {
    n := len(nums)
    count := make([]int, n+1)
    
    for _, num := range nums {
        count[num]++
        if count[num] > 1 {
            return num
        }
    }
    
    return -1
}
```

#### **Using Sorting**
```go
import "sort"

func findDuplicateSort(nums []int) int {
    sort.Ints(nums)
    
    for i := 1; i < len(nums); i++ {
        if nums[i] == nums[i-1] {
            return nums[i]
        }
    }
    
    return -1
}
```

#### **Using Mathematical Formula**
```go
func findDuplicateMath(nums []int) int {
    n := len(nums) - 1
    expectedSum := n * (n + 1) / 2
    actualSum := 0
    
    for _, num := range nums {
        actualSum += num
    }
    
    return actualSum - expectedSum
}
```

#### **Return All Duplicates**
```go
func findAllDuplicates(nums []int) []int {
    var result []int
    seen := make(map[int]bool)
    
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

### Complexity
- **Time Complexity:** O(n) for Floyd's algorithm, O(n log n) for binary search
- **Space Complexity:** O(1) for Floyd's algorithm, O(n) for hash set
