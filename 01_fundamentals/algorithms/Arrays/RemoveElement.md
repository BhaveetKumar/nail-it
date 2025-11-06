---
# Auto-generated front matter
Title: Removeelement
LastUpdated: 2025-11-06T20:45:58.725136
Tags: []
Status: draft
---

# Remove Element

### Problem
Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in-place. The order of the elements may be changed. Then return the number of elements in `nums` which are not equal to `val`.

Consider the number of elements in `nums` which are not equal to `val` be `k`, to get accepted, you need to do the following things:

- Change the array `nums` such that the first `k` elements of `nums` contain the elements which are not equal to `val`. The remaining elements of `nums` are not important as well as the size of `nums`.
- Return `k`.

**Example:**
```
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 3, 0, and 4.
```

### Golang Solution

```go
func removeElement(nums []int, val int) int {
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] != val {
            nums[k] = nums[i]
            k++
        }
    }
    
    return k
}
```

### Alternative Solutions

#### **Two Pointers from End**
```go
func removeElementTwoPointers(nums []int, val int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        if nums[left] == val {
            nums[left] = nums[right]
            right--
        } else {
            left++
        }
    }
    
    return left
}
```

#### **Using Swap**
```go
func removeElementSwap(nums []int, val int) int {
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] != val {
            if i != k {
                nums[k], nums[i] = nums[i], nums[k]
            }
            k++
        }
    }
    
    return k
}
```

#### **Return Removed Elements**
```go
func removeElementWithRemoved(nums []int, val int) (int, []int) {
    var removed []int
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] == val {
            removed = append(removed, nums[i])
        } else {
            nums[k] = nums[i]
            k++
        }
    }
    
    return k, removed
}
```

#### **Return Statistics**
```go
type RemoveStats struct {
    RemainingCount int
    RemovedCount   int
    RemainingElements []int
    RemovedElements   []int
}

func removeElementWithStats(nums []int, val int) RemoveStats {
    var remaining, removed []int
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if nums[i] == val {
            removed = append(removed, nums[i])
        } else {
            nums[k] = nums[i]
            remaining = append(remaining, nums[i])
            k++
        }
    }
    
    return RemoveStats{
        RemainingCount:   k,
        RemovedCount:     len(removed),
        RemainingElements: remaining,
        RemovedElements:   removed,
    }
}
```

#### **Remove Multiple Values**
```go
func removeMultipleElements(nums []int, vals []int) int {
    valSet := make(map[int]bool)
    for _, val := range vals {
        valSet[val] = true
    }
    
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if !valSet[nums[i]] {
            nums[k] = nums[i]
            k++
        }
    }
    
    return k
}
```

#### **Remove with Condition**
```go
func removeElementWithCondition(nums []int, condition func(int) bool) int {
    k := 0
    
    for i := 0; i < len(nums); i++ {
        if !condition(nums[i]) {
            nums[k] = nums[i]
            k++
        }
    }
    
    return k
}
```

### Complexity
- **Time Complexity:** O(n) where n is the length of the array
- **Space Complexity:** O(1) for in-place, O(n) for additional arrays
