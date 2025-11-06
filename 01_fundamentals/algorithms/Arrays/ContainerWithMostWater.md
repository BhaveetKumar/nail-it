---
# Auto-generated front matter
Title: Containerwithmostwater
LastUpdated: 2025-11-06T20:45:58.726850
Tags: []
Status: draft
---

# Container With Most Water

### Problem
You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `i-th` line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

**Example:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.
```

**Constraints:**
- n == height.length
- 2 ≤ n ≤ 10⁵
- 0 ≤ height[i] ≤ 10⁴

### Explanation

#### **Brute Force Approach**
- Check all possible pairs of lines
- Calculate area for each pair: `min(height[i], height[j]) * (j - i)`
- Keep track of maximum area
- Time Complexity: O(n²)
- Space Complexity: O(1)

#### **Two Pointers Approach**
- Start with two pointers at the beginning and end
- Calculate area and update maximum
- Move the pointer pointing to the shorter line
- This works because moving the longer line can only decrease the area
- Time Complexity: O(n)
- Space Complexity: O(1)

### Dry Run

**Input:** `height = [1,8,6,2,5,4,8,3,7]`

| Step | left | right | height[left] | height[right] | area | maxArea | Action |
|------|------|-------|--------------|---------------|------|---------|---------|
| 1 | 0 | 8 | 1 | 7 | min(1,7) * 8 = 8 | 8 | Move left |
| 2 | 1 | 8 | 8 | 7 | min(8,7) * 7 = 49 | 49 | Move right |
| 3 | 1 | 7 | 8 | 3 | min(8,3) * 6 = 18 | 49 | Move right |
| 4 | 1 | 6 | 8 | 8 | min(8,8) * 5 = 40 | 49 | Move left |
| 5 | 2 | 6 | 6 | 8 | min(6,8) * 4 = 24 | 49 | Move left |
| 6 | 3 | 6 | 2 | 8 | min(2,8) * 3 = 6 | 49 | Move left |
| 7 | 4 | 6 | 5 | 8 | min(5,8) * 2 = 10 | 49 | Move left |
| 8 | 5 | 6 | 4 | 8 | min(4,8) * 1 = 4 | 49 | Move left |

**Result:** `49` (between indices 1 and 8)

### Complexity
- **Time Complexity:** O(n) - Single pass with two pointers
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func maxArea(height []int) int {
    if len(height) < 2 {
        return 0
    }
    
    left, right := 0, len(height)-1
    maxArea := 0
    
    // Two pointers approach
    for left < right {
        // Calculate current area
        width := right - left
        currentArea := min(height[left], height[right]) * width
        
        // Update maximum area
        maxArea = max(maxArea, currentArea)
        
        // Move the pointer pointing to shorter line
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxArea
}

// Helper function to find minimum of two integers
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Helper function to find maximum of two integers
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Brute Force Implementation**
```go
func maxAreaBruteForce(height []int) int {
    maxArea := 0
    
    // Check all possible pairs
    for i := 0; i < len(height); i++ {
        for j := i + 1; j < len(height); j++ {
            width := j - i
            currentArea := min(height[i], height[j]) * width
            maxArea = max(maxArea, currentArea)
        }
    }
    
    return maxArea
}
```

#### **Optimized Two Pointers with Early Termination**
```go
func maxAreaOptimized(height []int) int {
    if len(height) < 2 {
        return 0
    }
    
    left, right := 0, len(height)-1
    maxArea := 0
    
    for left < right {
        width := right - left
        currentArea := min(height[left], height[right]) * width
        maxArea = max(maxArea, currentArea)
        
        // Early termination if we can't get better result
        if height[left] < height[right] {
            left++
            // Skip all lines shorter than current left
            for left < right && height[left] <= height[left-1] {
                left++
            }
        } else {
            right--
            // Skip all lines shorter than current right
            for left < right && height[right] <= height[right+1] {
                right--
            }
        }
    }
    
    return maxArea
}
```

### Notes / Variations

#### **Related Problems**
- **Trapping Rain Water**: Calculate trapped water between bars
- **Largest Rectangle in Histogram**: Find largest rectangle under histogram
- **Maximum Rectangle**: Find maximum rectangle in 2D matrix
- **Longest Mountain in Array**: Find longest mountain subarray

#### **ICPC Insights**
- **Two Pointers**: Classic technique for array problems
- **Greedy Strategy**: Moving shorter pointer is always optimal
- **Mathematical Proof**: Understand why the algorithm works
- **Edge Cases**: Handle arrays with duplicate heights

#### **Go-Specific Optimizations**
```go
// Use math.Max/Min for cleaner code (Go 1.21+)
import "math"

func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0
    
    for left < right {
        width := right - left
        currentArea := int(math.Min(float64(height[left]), float64(height[right]))) * width
        maxArea = int(math.Max(float64(maxArea), float64(currentArea)))
        
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxArea
}
```

#### **Real-World Applications**
- **Water Management**: Calculate maximum water storage capacity
- **Architecture**: Design optimal container shapes
- **Game Development**: Calculate collision boundaries
- **Data Visualization**: Optimize chart layouts

### Testing

```go
func TestMaxArea(t *testing.T) {
    tests := []struct {
        height   []int
        expected int
    }{
        {[]int{1, 8, 6, 2, 5, 4, 8, 3, 7}, 49},
        {[]int{1, 1}, 1},
        {[]int{4, 3, 2, 1, 4}, 16},
        {[]int{1, 2, 1}, 2},
        {[]int{2, 3, 4, 5, 18, 17, 6}, 17},
    }
    
    for _, test := range tests {
        result := maxArea(test.height)
        if result != test.expected {
            t.Errorf("maxArea(%v) = %d, expected %d", 
                test.height, result, test.expected)
        }
    }
}
```

### Visualization

```
Height: [1, 8, 6, 2, 5, 4, 8, 3, 7]
Index:   0  1  2  3  4  5  6  7  8

Step 1: left=0, right=8
       1                   7
       |                   |
       |                   |
       |                   |
       |                   |
       |                   |
       |                   |
       |                   |
       |                   |
       └───────────────────┘
       Area = min(1,7) * 8 = 8

Step 2: left=1, right=8
           8               7
           |               |
           |               |
           |               |
           |               |
           |               |
           |               |
           |               |
           |               |
           └───────────────┘
           Area = min(8,7) * 7 = 49 ← Maximum!

Step 3: left=1, right=7
           8           3
           |           |
           |           |
           |           |
           |           |
           |           |
           |           |
           |           |
           |           |
           └───────────┘
           Area = min(8,3) * 6 = 18
```

### Mathematical Proof

**Why does moving the shorter pointer work?**

Let's say we have two pointers at positions `i` and `j` where `height[i] < height[j]`.

- Current area = `height[i] * (j - i)`
- If we move the right pointer (taller line) to `j-1`:
  - New area = `min(height[i], height[j-1]) * (j-1-i)`
  - Since `height[i]` is the shorter line, `min(height[i], height[j-1]) ≤ height[i]`
  - And `(j-1-i) < (j-i)`
  - Therefore, new area ≤ current area

- If we move the left pointer (shorter line) to `i+1`:
  - New area = `min(height[i+1], height[j]) * (j-(i+1))`
  - This could potentially be larger if `height[i+1] > height[i]`

**Conclusion:** Moving the shorter pointer is the only way to potentially increase the area.
