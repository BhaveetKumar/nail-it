# Single Number

### Problem
Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example:**
```
Input: nums = [2,2,1]
Output: 1

Input: nums = [4,1,2,1,2]
Output: 4

Input: nums = [1]
Output: 1
```

### Golang Solution

```go
func singleNumber(nums []int) int {
    result := 0
    
    for _, num := range nums {
        result ^= num
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Hash Map**
```go
func singleNumberHashMap(nums []int) int {
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

#### **Using Set**
```go
func singleNumberSet(nums []int) int {
    seen := make(map[int]bool)
    
    for _, num := range nums {
        if seen[num] {
            delete(seen, num)
        } else {
            seen[num] = true
        }
    }
    
    for num := range seen {
        return num
    }
    
    return -1
}
```

#### **Using Math**
```go
func singleNumberMath(nums []int) int {
    sum := 0
    uniqueSum := 0
    seen := make(map[int]bool)
    
    for _, num := range nums {
        sum += num
        if !seen[num] {
            uniqueSum += num
            seen[num] = true
        }
    }
    
    return 2*uniqueSum - sum
}
```

#### **Using Sorting**
```go
import "sort"

func singleNumberSort(nums []int) int {
    sort.Ints(nums)
    
    for i := 0; i < len(nums)-1; i += 2 {
        if nums[i] != nums[i+1] {
            return nums[i]
        }
    }
    
    return nums[len(nums)-1]
}
```

#### **Using Array**
```go
func singleNumberArray(nums []int) int {
    count := make([]int, 60001) // Assuming range [-30000, 30000]
    
    for _, num := range nums {
        count[num+30000]++
    }
    
    for i, freq := range count {
        if freq == 1 {
            return i - 30000
        }
    }
    
    return -1
}
```

#### **Return All Single Numbers**
```go
func findAllSingleNumbers(nums []int) []int {
    count := make(map[int]int)
    
    for _, num := range nums {
        count[num]++
    }
    
    var result []int
    for num, freq := range count {
        if freq == 1 {
            result = append(result, num)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for XOR, O(n log n) for sorting
- **Space Complexity:** O(1) for XOR, O(n) for hash map