---
# Auto-generated front matter
Title: Threesumclosest
LastUpdated: 2025-11-06T20:45:58.722310
Tags: []
Status: draft
---

# 3Sum Closest

### Problem
Given an integer array `nums` of length `n` and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers.

You may assume that each input would have exactly one solution.

**Example:**
```
Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

Input: nums = [0,0,0], target = 1
Output: 0
```

### Golang Solution

```go
import "sort"

func threeSumClosest(nums []int, target int) int {
    sort.Ints(nums)
    n := len(nums)
    closest := nums[0] + nums[1] + nums[2]
    
    for i := 0; i < n-2; i++ {
        left, right := i+1, n-1
        
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            
            if abs(sum-target) < abs(closest-target) {
                closest = sum
            }
            
            if sum < target {
                left++
            } else if sum > target {
                right--
            } else {
                return sum
            }
        }
    }
    
    return closest
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```

### Alternative Solutions

#### **Brute Force**
```go
func threeSumClosestBruteForce(nums []int, target int) int {
    n := len(nums)
    closest := nums[0] + nums[1] + nums[2]
    
    for i := 0; i < n-2; i++ {
        for j := i + 1; j < n-1; j++ {
            for k := j + 1; k < n; k++ {
                sum := nums[i] + nums[j] + nums[k]
                if abs(sum-target) < abs(closest-target) {
                    closest = sum
                }
            }
        }
    }
    
    return closest
}
```

#### **Return with Indices**
```go
type ThreeSumResult struct {
    Sum     int
    Indices []int
    Diff    int
}

func threeSumClosestWithIndices(nums []int, target int) ThreeSumResult {
    sort.Ints(nums)
    n := len(nums)
    closest := nums[0] + nums[1] + nums[2]
    bestIndices := []int{0, 1, 2}
    
    for i := 0; i < n-2; i++ {
        left, right := i+1, n-1
        
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            
            if abs(sum-target) < abs(closest-target) {
                closest = sum
                bestIndices = []int{i, left, right}
            }
            
            if sum < target {
                left++
            } else if sum > target {
                right--
            } else {
                return ThreeSumResult{
                    Sum:     sum,
                    Indices: []int{i, left, right},
                    Diff:    0,
                }
            }
        }
    }
    
    return ThreeSumResult{
        Sum:     closest,
        Indices: bestIndices,
        Diff:    abs(closest - target),
    }
}
```

#### **Return All Closest Sums**
```go
func allClosestSums(nums []int, target int) []int {
    sort.Ints(nums)
    n := len(nums)
    var results []int
    minDiff := math.MaxInt32
    
    for i := 0; i < n-2; i++ {
        left, right := i+1, n-1
        
        for left < right {
            sum := nums[i] + nums[left] + nums[right]
            diff := abs(sum - target)
            
            if diff < minDiff {
                minDiff = diff
                results = []int{sum}
            } else if diff == minDiff {
                results = append(results, sum)
            }
            
            if sum < target {
                left++
            } else if sum > target {
                right--
            } else {
                return []int{sum}
            }
        }
    }
    
    return results
}
```

#### **Return Statistics**
```go
type ThreeSumStats struct {
    ClosestSum    int
    MinDiff       int
    MaxDiff       int
    AvgDiff       float64
    TotalCombinations int
}

func threeSumStatistics(nums []int, target int) ThreeSumStats {
    n := len(nums)
    var diffs []int
    closest := nums[0] + nums[1] + nums[2]
    minDiff := abs(closest - target)
    maxDiff := 0
    totalCombinations := 0
    
    for i := 0; i < n-2; i++ {
        for j := i + 1; j < n-1; j++ {
            for k := j + 1; k < n; k++ {
                sum := nums[i] + nums[j] + nums[k]
                diff := abs(sum - target)
                diffs = append(diffs, diff)
                totalCombinations++
                
                if diff < minDiff {
                    minDiff = diff
                    closest = sum
                }
                
                if diff > maxDiff {
                    maxDiff = diff
                }
            }
        }
    }
    
    sum := 0
    for _, diff := range diffs {
        sum += diff
    }
    
    return ThreeSumStats{
        ClosestSum:       closest,
        MinDiff:          minDiff,
        MaxDiff:          maxDiff,
        AvgDiff:          float64(sum) / float64(len(diffs)),
        TotalCombinations: totalCombinations,
    }
}
```

### Complexity
- **Time Complexity:** O(n²) for optimized, O(n³) for brute force
- **Space Complexity:** O(1) for in-place, O(n) for additional arrays
