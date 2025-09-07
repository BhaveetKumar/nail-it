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

### Alternative Solutions

#### **Brute Force**
```go
func maxAreaBruteForce(height []int) int {
    maxWater := 0
    
    for i := 0; i < len(height); i++ {
        for j := i + 1; j < len(height); j++ {
            width := j - i
            minHeight := min(height[i], height[j])
            area := width * minHeight
            maxWater = max(maxWater, area)
        }
    }
    
    return maxWater
}
```

### Complexity
- **Time Complexity:** O(n) for two pointers, O(n²) for brute force
- **Space Complexity:** O(1)
