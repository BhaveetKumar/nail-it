---
# Auto-generated front matter
Title: Trappingrainwater
LastUpdated: 2025-11-06T20:45:58.725769
Tags: []
Status: draft
---

# Trapping Rain Water

### Problem
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

**Example:**
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
```

### Golang Solution

```go
func trap(height []int) int {
    if len(height) < 3 {
        return 0
    }
    
    left, right := 0, len(height)-1
    leftMax, rightMax := 0, 0
    water := 0
    
    for left < right {
        if height[left] < height[right] {
            if height[left] >= leftMax {
                leftMax = height[left]
            } else {
                water += leftMax - height[left]
            }
            left++
        } else {
            if height[right] >= rightMax {
                rightMax = height[right]
            } else {
                water += rightMax - height[right]
            }
            right--
        }
    }
    
    return water
}
```

### Alternative Solutions

#### **Stack Approach**
```go
func trapStack(height []int) int {
    stack := []int{}
    water := 0
    
    for i := 0; i < len(height); i++ {
        for len(stack) > 0 && height[i] > height[stack[len(stack)-1]] {
            top := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            
            if len(stack) == 0 {
                break
            }
            
            distance := i - stack[len(stack)-1] - 1
            boundedHeight := min(height[i], height[stack[len(stack)-1]]) - height[top]
            water += distance * boundedHeight
        }
        stack = append(stack, i)
    }
    
    return water
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for two pointers, O(n) for stack
