---
# Auto-generated front matter
Title: Findpeakelement
LastUpdated: 2025-11-06T20:45:58.721653
Tags: []
Status: draft
---

# Find Peak Element

### Problem
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that `nums[-1] = nums[n] = -∞`.

You must write an algorithm that runs in O(log n) time.

**Example:**
```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
```

### Golang Solution

```go
func findPeakElement(nums []int) int {
    left, right := 0, len(nums)-1
    
    for left < right {
        mid := left + (right-left)/2
        
        if nums[mid] > nums[mid+1] {
            right = mid
        } else {
            left = mid + 1
        }
    }
    
    return left
}
```

### Alternative Solutions

#### **Linear Search**
```go
func findPeakElementLinear(nums []int) int {
    for i := 0; i < len(nums)-1; i++ {
        if nums[i] > nums[i+1] {
            return i
        }
    }
    return len(nums) - 1
}
```

#### **Recursive Binary Search**
```go
func findPeakElementRecursive(nums []int) int {
    return search(nums, 0, len(nums)-1)
}

func search(nums []int, left, right int) int {
    if left == right {
        return left
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] > nums[mid+1] {
        return search(nums, left, mid)
    }
    
    return search(nums, mid+1, right)
}
```

#### **Find All Peaks**
```go
func findAllPeakElements(nums []int) []int {
    var peaks []int
    
    for i := 0; i < len(nums); i++ {
        isPeak := true
        
        if i > 0 && nums[i] <= nums[i-1] {
            isPeak = false
        }
        
        if i < len(nums)-1 && nums[i] <= nums[i+1] {
            isPeak = false
        }
        
        if isPeak {
            peaks = append(peaks, i)
        }
    }
    
    return peaks
}
```

#### **Find Global Maximum**
```go
func findPeakElementGlobalMax(nums []int) int {
    maxIndex := 0
    
    for i := 1; i < len(nums); i++ {
        if nums[i] > nums[maxIndex] {
            maxIndex = i
        }
    }
    
    return maxIndex
}
```

### Complexity
- **Time Complexity:** O(log n) for binary search, O(n) for linear search
- **Space Complexity:** O(1) for iterative, O(log n) for recursive