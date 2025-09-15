# Single Number II

### Problem
Given an integer array `nums` where every element appears three times except for one, which appears exactly once. Find the single element and return it.

You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example:**
```
Input: nums = [2,2,3,2]
Output: 3

Input: nums = [0,1,0,1,0,1,99]
Output: 99
```

### Golang Solution

```go
func singleNumber(nums []int) int {
    ones, twos := 0, 0
    
    for _, num := range nums {
        ones = (ones ^ num) & ^twos
        twos = (twos ^ num) & ^ones
    }
    
    return ones
}
```

### Alternative Solutions

#### **Using Map**
```go
func singleNumberMap(nums []int) int {
    count := make(map[int]int)
    
    for _, num := range nums {
        count[num]++
    }
    
    for num, freq := range count {
        if freq == 1 {
            return num
        }
    }
    
    return -1
}
```

#### **Bit Manipulation with Array**
```go
func singleNumberBits(nums []int) int {
    result := 0
    
    for i := 0; i < 32; i++ {
        sum := 0
        for _, num := range nums {
            if (num>>i)&1 == 1 {
                sum++
            }
        }
        
        if sum%3 != 0 {
            result |= (1 << i)
        }
    }
    
    return result
}
```

#### **Sorting Approach**
```go
import "sort"

func singleNumberSort(nums []int) int {
    sort.Ints(nums)
    
    for i := 0; i < len(nums); i += 3 {
        if i+1 >= len(nums) || nums[i] != nums[i+1] {
            return nums[i]
        }
    }
    
    return -1
}
```

### Complexity
- **Time Complexity:** O(n) for bit manipulation, O(n log n) for sorting
- **Space Complexity:** O(1) for bit manipulation, O(n) for map
