---
# Auto-generated front matter
Title: Medianoftwosortedarrays
LastUpdated: 2025-11-06T20:45:58.725009
Tags: []
Status: draft
---

# Median of Two Sorted Arrays

### Problem
Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays.

The overall run time complexity should be `O(log (m+n))`.

**Example:**
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
```

### Golang Solution

```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
    if len(nums1) > len(nums2) {
        nums1, nums2 = nums2, nums1
    }
    
    m, n := len(nums1), len(nums2)
    left, right := 0, m
    
    for left <= right {
        partitionX := (left + right) / 2
        partitionY := (m + n + 1) / 2 - partitionX
        
        maxLeftX := math.MinInt64
        if partitionX > 0 {
            maxLeftX = nums1[partitionX-1]
        }
        
        minRightX := math.MaxInt64
        if partitionX < m {
            minRightX = nums1[partitionX]
        }
        
        maxLeftY := math.MinInt64
        if partitionY > 0 {
            maxLeftY = nums2[partitionY-1]
        }
        
        minRightY := math.MaxInt64
        if partitionY < n {
            minRightY = nums2[partitionY]
        }
        
        if maxLeftX <= minRightY && maxLeftY <= minRightX {
            if (m+n)%2 == 0 {
                return float64(max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2.0
            } else {
                return float64(max(maxLeftX, maxLeftY))
            }
        } else if maxLeftX > minRightY {
            right = partitionX - 1
        } else {
            left = partitionX + 1
        }
    }
    
    return 0.0
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(log(min(m, n)))
- **Space Complexity:** O(1)
