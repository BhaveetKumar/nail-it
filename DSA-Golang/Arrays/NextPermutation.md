# Next Permutation

### Problem
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

The next permutation of an array of integers is the next lexicographically greater permutation of its integer.

**Example:**
```
Input: nums = [1,2,3]
Output: [1,3,2]

Input: nums = [3,2,1]
Output: [1,2,3]

Input: nums = [1,1,5]
Output: [1,5,1]
```

### Golang Solution

```go
func nextPermutation(nums []int) {
    n := len(nums)
    i := n - 2
    
    // Find the largest index i such that nums[i] < nums[i+1]
    for i >= 0 && nums[i] >= nums[i+1] {
        i--
    }
    
    if i >= 0 {
        // Find the largest index j such that nums[i] < nums[j]
        j := n - 1
        for nums[j] <= nums[i] {
            j--
        }
        
        // Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]
    }
    
    // Reverse the suffix starting at nums[i+1]
    reverse(nums, i+1, n-1)
}

func reverse(nums []int, start, end int) {
    for start < end {
        nums[start], nums[end] = nums[end], nums[start]
        start++
        end--
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
