# Missing Number

### Problem
Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the only number in the range that is missing from the array.

**Example:**
```
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
```

### Golang Solution

```go
func missingNumber(nums []int) int {
    n := len(nums)
    expectedSum := n * (n + 1) / 2
    actualSum := 0
    
    for _, num := range nums {
        actualSum += num
    }
    
    return expectedSum - actualSum
}
```

### Alternative Solutions

#### **XOR Approach**
```go
func missingNumberXOR(nums []int) int {
    result := len(nums)
    
    for i := 0; i < len(nums); i++ {
        result ^= i ^ nums[i]
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
