# Remove Duplicates from Sorted Array

### Problem
Given an integer array `nums` sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array `nums`. More formally, if there are `k` elements after removing the duplicates, then the first `k` elements of `nums` should hold the final result. It does not matter what you leave beyond the first `k` elements.

Return `k` after placing the final result in the first `k` slots of `nums`.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

**Example:**
```
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
```

### Golang Solution

```go
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    writeIndex := 1
    
    for readIndex := 1; readIndex < len(nums); readIndex++ {
        if nums[readIndex] != nums[readIndex-1] {
            nums[writeIndex] = nums[readIndex]
            writeIndex++
        }
    }
    
    return writeIndex
}
```

### Alternative Solutions

#### **Using Two Pointers Explicitly**
```go
func removeDuplicatesTwoPointers(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    left, right := 0, 1
    
    for right < len(nums) {
        if nums[right] != nums[left] {
            left++
            nums[left] = nums[right]
        }
        right++
    }
    
    return left + 1
}
```

#### **Using Map (Not In-Place)**
```go
func removeDuplicatesMap(nums []int) int {
    seen := make(map[int]bool)
    var result []int
    
    for _, num := range nums {
        if !seen[num] {
            seen[num] = true
            result = append(result, num)
        }
    }
    
    // Copy back to original array
    for i := 0; i < len(result); i++ {
        nums[i] = result[i]
    }
    
    return len(result)
}
```

#### **Using Set (Not In-Place)**
```go
func removeDuplicatesSet(nums []int) int {
    seen := make(map[int]bool)
    var result []int
    
    for _, num := range nums {
        if !seen[num] {
            seen[num] = true
            result = append(result, num)
        }
    }
    
    // Copy back to original array
    copy(nums, result)
    
    return len(result)
}
```

#### **Recursive Approach**
```go
func removeDuplicatesRecursive(nums []int) int {
    if len(nums) <= 1 {
        return len(nums)
    }
    
    if nums[0] == nums[1] {
        return removeDuplicatesRecursive(nums[1:])
    }
    
    return 1 + removeDuplicatesRecursive(nums[1:])
}
```

#### **Return Modified Array**
```go
func removeDuplicatesReturn(nums []int) []int {
    if len(nums) == 0 {
        return nums
    }
    
    writeIndex := 1
    
    for readIndex := 1; readIndex < len(nums); readIndex++ {
        if nums[readIndex] != nums[readIndex-1] {
            nums[writeIndex] = nums[readIndex]
            writeIndex++
        }
    }
    
    return nums[:writeIndex]
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for in-place, O(n) for map/set approaches
