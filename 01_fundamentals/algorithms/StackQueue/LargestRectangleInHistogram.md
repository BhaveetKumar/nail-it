---
# Auto-generated front matter
Title: Largestrectangleinhistogram
LastUpdated: 2025-11-06T20:45:58.704633
Tags: []
Status: draft
---

# Largest Rectangle in Histogram

### Problem
Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Example:**
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.

Input: heights = [2,4]
Output: 4
```

### Golang Solution

```go
func largestRectangleArea(heights []int) int {
    stack := []int{}
    maxArea := 0
    
    for i := 0; i <= len(heights); i++ {
        var h int
        if i == len(heights) {
            h = 0
        } else {
            h = heights[i]
        }
        
        for len(stack) > 0 && h < heights[stack[len(stack)-1]] {
            height := heights[stack[len(stack)-1]]
            stack = stack[:len(stack)-1]
            
            var width int
            if len(stack) == 0 {
                width = i
            } else {
                width = i - stack[len(stack)-1] - 1
            }
            
            area := height * width
            if area > maxArea {
                maxArea = area
            }
        }
        
        stack = append(stack, i)
    }
    
    return maxArea
}
```

### Alternative Solutions

#### **Brute Force**
```go
func largestRectangleAreaBruteForce(heights []int) int {
    maxArea := 0
    
    for i := 0; i < len(heights); i++ {
        minHeight := heights[i]
        
        for j := i; j < len(heights); j++ {
            if heights[j] < minHeight {
                minHeight = heights[j]
            }
            
            area := minHeight * (j - i + 1)
            if area > maxArea {
                maxArea = area
            }
        }
    }
    
    return maxArea
}
```

#### **Divide and Conquer**
```go
func largestRectangleAreaDivideConquer(heights []int) int {
    return divideConquer(heights, 0, len(heights)-1)
}

func divideConquer(heights []int, left, right int) int {
    if left > right {
        return 0
    }
    
    if left == right {
        return heights[left]
    }
    
    // Find minimum height index
    minIndex := left
    for i := left + 1; i <= right; i++ {
        if heights[i] < heights[minIndex] {
            minIndex = i
        }
    }
    
    // Calculate area with minimum height
    areaWithMin := heights[minIndex] * (right - left + 1)
    
    // Recursively find maximum in left and right subarrays
    leftArea := divideConquer(heights, left, minIndex-1)
    rightArea := divideConquer(heights, minIndex+1, right)
    
    return max(areaWithMin, max(leftArea, rightArea))
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

#### **Using Two Arrays**
```go
func largestRectangleAreaTwoArrays(heights []int) int {
    n := len(heights)
    left := make([]int, n)
    right := make([]int, n)
    
    // Find next smaller element on the left
    stack := []int{}
    for i := 0; i < n; i++ {
        for len(stack) > 0 && heights[stack[len(stack)-1]] >= heights[i] {
            stack = stack[:len(stack)-1]
        }
        
        if len(stack) == 0 {
            left[i] = -1
        } else {
            left[i] = stack[len(stack)-1]
        }
        
        stack = append(stack, i)
    }
    
    // Find next smaller element on the right
    stack = []int{}
    for i := n - 1; i >= 0; i-- {
        for len(stack) > 0 && heights[stack[len(stack)-1]] >= heights[i] {
            stack = stack[:len(stack)-1]
        }
        
        if len(stack) == 0 {
            right[i] = n
        } else {
            right[i] = stack[len(stack)-1]
        }
        
        stack = append(stack, i)
    }
    
    // Calculate maximum area
    maxArea := 0
    for i := 0; i < n; i++ {
        area := heights[i] * (right[i] - left[i] - 1)
        if area > maxArea {
            maxArea = area
        }
    }
    
    return maxArea
}
```

#### **Optimized Stack**
```go
func largestRectangleAreaOptimized(heights []int) int {
    stack := []int{}
    maxArea := 0
    
    for i := 0; i < len(heights); i++ {
        for len(stack) > 0 && heights[stack[len(stack)-1]] > heights[i] {
            height := heights[stack[len(stack)-1]]
            stack = stack[:len(stack)-1]
            
            var width int
            if len(stack) == 0 {
                width = i
            } else {
                width = i - stack[len(stack)-1] - 1
            }
            
            area := height * width
            if area > maxArea {
                maxArea = area
            }
        }
        
        stack = append(stack, i)
    }
    
    // Process remaining elements in stack
    for len(stack) > 0 {
        height := heights[stack[len(stack)-1]]
        stack = stack[:len(stack)-1]
        
        var width int
        if len(stack) == 0 {
            width = len(heights)
        } else {
            width = len(heights) - stack[len(stack)-1] - 1
        }
        
        area := height * width
        if area > maxArea {
            maxArea = area
        }
    }
    
    return maxArea
}
```

### Complexity
- **Time Complexity:** O(n) for stack, O(n²) for brute force, O(n log n) for divide and conquer
- **Space Complexity:** O(n)