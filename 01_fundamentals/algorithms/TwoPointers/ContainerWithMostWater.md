---
# Auto-generated front matter
Title: Containerwithmostwater
LastUpdated: 2025-11-06T20:45:58.700129
Tags: []
Status: draft
---

# Container With Most Water

### Problem
You are given an integer array `height` of length `n`. There are `n` vertical lines drawn such that the two endpoints of the `ith` line are `(i, 0)` and `(i, height[i])`.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

**Example:**
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

Input: height = [1,1]
Output: 1
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
        
        if currentArea > maxWater {
            maxWater = currentArea
        }
        
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
            currentArea := width * minHeight
            
            if currentArea > maxWater {
                maxWater = currentArea
            }
        }
    }
    
    return maxWater
}
```

#### **Return with Indices**
```go
func maxAreaWithIndices(height []int) (int, int, int) {
    left, right := 0, len(height)-1
    maxWater := 0
    bestLeft, bestRight := 0, 0
    
    for left < right {
        width := right - left
        minHeight := min(height[left], height[right])
        currentArea := width * minHeight
        
        if currentArea > maxWater {
            maxWater = currentArea
            bestLeft = left
            bestRight = right
        }
        
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }
    
    return maxWater, bestLeft, bestRight
}
```

#### **Return All Areas**
```go
func allAreas(height []int) []int {
    var areas []int
    
    for i := 0; i < len(height); i++ {
        for j := i + 1; j < len(height); j++ {
            width := j - i
            minHeight := min(height[i], height[j])
            currentArea := width * minHeight
            areas = append(areas, currentArea)
        }
    }
    
    return areas
}
```

#### **Return Area Statistics**
```go
type AreaStats struct {
    MaxArea    int
    MinArea    int
    AvgArea    float64
    TotalPairs int
    MaxIndices []int
}

func areaStatistics(height []int) AreaStats {
    if len(height) < 2 {
        return AreaStats{}
    }
    
    maxArea := 0
    minArea := math.MaxInt32
    sum := 0
    totalPairs := 0
    var maxIndices []int
    
    for i := 0; i < len(height); i++ {
        for j := i + 1; j < len(height); j++ {
            width := j - i
            minHeight := min(height[i], height[j])
            currentArea := width * minHeight
            
            if currentArea > maxArea {
                maxArea = currentArea
                maxIndices = []int{i, j}
            }
            
            if currentArea < minArea {
                minArea = currentArea
            }
            
            sum += currentArea
            totalPairs++
        }
    }
    
    return AreaStats{
        MaxArea:    maxArea,
        MinArea:    minArea,
        AvgArea:    float64(sum) / float64(totalPairs),
        TotalPairs: totalPairs,
        MaxIndices: maxIndices,
    }
}
```

#### **Return All Valid Containers**
```go
type Container struct {
    LeftIndex  int
    RightIndex int
    Area       int
    Width      int
    MinHeight  int
}

func allContainers(height []int) []Container {
    var containers []Container
    
    for i := 0; i < len(height); i++ {
        for j := i + 1; j < len(height); j++ {
            width := j - i
            minHeight := min(height[i], height[j])
            currentArea := width * minHeight
            
            containers = append(containers, Container{
                LeftIndex:  i,
                RightIndex: j,
                Area:       currentArea,
                Width:      width,
                MinHeight:  minHeight,
            })
        }
    }
    
    return containers
}
```

#### **Return Sorted Containers**
```go
import "sort"

func sortedContainers(height []int) []Container {
    containers := allContainers(height)
    
    sort.Slice(containers, func(i, j int) bool {
        return containers[i].Area > containers[j].Area
    })
    
    return containers
}
```

### Complexity
- **Time Complexity:** O(n) for two pointers, O(n²) for brute force
- **Space Complexity:** O(1) for two pointers, O(n²) for all containers