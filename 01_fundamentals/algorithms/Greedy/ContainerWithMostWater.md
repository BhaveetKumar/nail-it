---
# Auto-generated front matter
Title: Containerwithmostwater
LastUpdated: 2025-11-06T20:45:58.731394
Tags: []
Status: draft
---

# Container With Most Water

### Problem
You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `ith` line are `(i, 0)` and `(i, height[i])`.

Find two lines that, together with the x-axis forms a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Example:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```

### Golang Solution

```go
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxWater := 0
    
    for left < right {
        width := right - left
        minHeight := min(height[left], height[right])
        currentArea := width * minHeight
        maxWater = max(maxWater, currentArea)
        
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxWater
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
