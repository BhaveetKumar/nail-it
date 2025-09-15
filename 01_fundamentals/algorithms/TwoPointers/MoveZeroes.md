# Move Zeroes

### Problem
Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

**Example:**
```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]
```

### Golang Solution

```go
func moveZeroes(nums []int) {
    writeIndex := 0
    
    // Move all non-zero elements to the front
    for readIndex := 0; readIndex < len(nums); readIndex++ {
        if nums[readIndex] != 0 {
            nums[writeIndex] = nums[readIndex]
            writeIndex++
        }
    }
    
    // Fill remaining positions with zeros
    for writeIndex < len(nums) {
        nums[writeIndex] = 0
        writeIndex++
    }
}
```

### Alternative Solutions

#### **Two Pointers Approach**
```go
func moveZeroesTwoPointers(nums []int) {
    left, right := 0, 0
    
    for right < len(nums) {
        if nums[right] != 0 {
            nums[left], nums[right] = nums[right], nums[left]
            left++
        }
        right++
    }
}
```

#### **Using Swap**
```go
func moveZeroesSwap(nums []int) {
    nonZeroIndex := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] != 0 {
            if i != nonZeroIndex {
                nums[i], nums[nonZeroIndex] = nums[nonZeroIndex], nums[i]
            }
            nonZeroIndex++
        }
    }
}
```

#### **Using Filter and Append**
```go
func moveZeroesFilter(nums []int) {
    var nonZeros []int
    var zeros []int
    
    for _, num := range nums {
        if num == 0 {
            zeros = append(zeros, num)
        } else {
            nonZeros = append(nonZeros, num)
        }
    }
    
    // Copy back to original array
    copy(nums, append(nonZeros, zeros...))
}
```

#### **Using Count**
```go
func moveZeroesCount(nums []int) {
    zeroCount := 0
    
    // Count zeros and move non-zeros
    for i := 0; i < len(nums); i++ {
        if nums[i] == 0 {
            zeroCount++
        } else {
            nums[i-zeroCount] = nums[i]
        }
    }
    
    // Fill end with zeros
    for i := len(nums) - zeroCount; i < len(nums); i++ {
        nums[i] = 0
    }
}
```

#### **Return Modified Array**
```go
func moveZeroesReturn(nums []int) []int {
    writeIndex := 0
    
    // Move all non-zero elements to the front
    for readIndex := 0; readIndex < len(nums); readIndex++ {
        if nums[readIndex] != 0 {
            nums[writeIndex] = nums[readIndex]
            writeIndex++
        }
    }
    
    // Fill remaining positions with zeros
    for writeIndex < len(nums) {
        nums[writeIndex] = 0
        writeIndex++
    }
    
    return nums
}
```

#### **Move to Beginning**
```go
func moveZeroesToBeginning(nums []int) {
    writeIndex := len(nums) - 1
    
    // Move all non-zero elements to the end
    for readIndex := len(nums) - 1; readIndex >= 0; readIndex-- {
        if nums[readIndex] != 0 {
            nums[writeIndex] = nums[readIndex]
            writeIndex--
        }
    }
    
    // Fill beginning with zeros
    for writeIndex >= 0 {
        nums[writeIndex] = 0
        writeIndex--
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
