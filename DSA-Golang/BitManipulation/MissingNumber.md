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

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
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

#### **Using XOR**
```go
func missingNumberXOR(nums []int) int {
    result := len(nums)
    
    for i, num := range nums {
        result ^= i ^ num
    }
    
    return result
}
```

#### **Using Hash Set**
```go
func missingNumberHashSet(nums []int) int {
    numSet := make(map[int]bool)
    
    for _, num := range nums {
        numSet[num] = true
    }
    
    for i := 0; i <= len(nums); i++ {
        if !numSet[i] {
            return i
        }
    }
    
    return -1
}
```

#### **Using Sorting**
```go
import "sort"

func missingNumberSort(nums []int) int {
    sort.Ints(nums)
    
    for i := 0; i < len(nums); i++ {
        if nums[i] != i {
            return i
        }
    }
    
    return len(nums)
}
```

#### **Using Array as Hash Map**
```go
func missingNumberArray(nums []int) int {
    n := len(nums)
    present := make([]bool, n+1)
    
    for _, num := range nums {
        present[num] = true
    }
    
    for i := 0; i <= n; i++ {
        if !present[i] {
            return i
        }
    }
    
    return -1
}
```

#### **Using Binary Search**
```go
import "sort"

func missingNumberBinarySearch(nums []int) int {
    sort.Ints(nums)
    
    left, right := 0, len(nums)
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] == mid {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}
```

#### **Return All Missing Numbers**
```go
func findAllMissingNumbers(nums []int) []int {
    n := len(nums)
    present := make([]bool, n+1)
    var result []int
    
    for _, num := range nums {
        present[num] = true
    }
    
    for i := 0; i <= n; i++ {
        if !present[i] {
            result = append(result, i)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for most approaches, O(n log n) for sorting
- **Space Complexity:** O(1) for XOR/math, O(n) for hash set