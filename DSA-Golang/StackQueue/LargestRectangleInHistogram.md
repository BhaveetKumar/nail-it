# Largest Rectangle in Histogram

### Problem
Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Example:**
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
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
    
    minIndex := left
    for i := left; i <= right; i++ {
        if heights[i] < heights[minIndex] {
            minIndex = i
        }
    }
    
    area := heights[minIndex] * (right - left + 1)
    leftArea := divideConquer(heights, left, minIndex-1)
    rightArea := divideConquer(heights, minIndex+1, right)
    
    return max(area, max(leftArea, rightArea))
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n) for stack, O(n log n) for divide and conquer
- **Space Complexity:** O(n)
