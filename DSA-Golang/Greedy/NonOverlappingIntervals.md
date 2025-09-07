# Non-overlapping Intervals

### Problem
Given an array of intervals `intervals` where `intervals[i] = [starti, endi]`, return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

**Example:**
```
Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
```

### Golang Solution

```go
import "sort"

func eraseOverlapIntervals(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    // Sort by end time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][1] < intervals[j][1]
    })
    
    count := 1
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] >= end {
            count++
            end = intervals[i][1]
        }
    }
    
    return len(intervals) - count
}
```

### Alternative Solutions

#### **Sort by Start Time**
```go
import "sort"

func eraseOverlapIntervalsStart(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    // Sort by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    count := 1
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] >= end {
            count++
            end = intervals[i][1]
        } else {
            // Keep the interval with smaller end time
            if intervals[i][1] < end {
                end = intervals[i][1]
            }
        }
    }
    
    return len(intervals) - count
}
```

#### **Return Removed Intervals**
```go
import "sort"

func eraseOverlapIntervalsWithRemoved(intervals [][]int) (int, [][]int) {
    if len(intervals) == 0 {
        return 0, [][]int{}
    }
    
    // Sort by end time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][1] < intervals[j][1]
    })
    
    var kept [][]int
    var removed [][]int
    
    kept = append(kept, intervals[0])
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] >= end {
            kept = append(kept, intervals[i])
            end = intervals[i][1]
        } else {
            removed = append(removed, intervals[i])
        }
    }
    
    return len(removed), removed
}
```

#### **Return Maximum Non-overlapping Intervals**
```go
import "sort"

func maxNonOverlappingIntervals(intervals [][]int) [][]int {
    if len(intervals) == 0 {
        return [][]int{}
    }
    
    // Sort by end time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][1] < intervals[j][1]
    })
    
    var result [][]int
    result = append(result, intervals[0])
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] >= end {
            result = append(result, intervals[i])
            end = intervals[i][1]
        }
    }
    
    return result
}
```

#### **Return All Possible Solutions**
```go
import "sort"

func allNonOverlappingSolutions(intervals [][]int) [][][]int {
    if len(intervals) == 0 {
        return [][][]int{}
    }
    
    // Sort by end time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][1] < intervals[j][1]
    })
    
    var result [][][]int
    
    var backtrack func(int, []int, [][]int)
    backtrack = func(index int, current []int, selected [][]int) {
        if index == len(intervals) {
            if len(selected) > 0 {
                solution := make([][]int, len(selected))
                copy(solution, selected)
                result = append(result, solution)
            }
            return
        }
        
        // Try not selecting current interval
        backtrack(index+1, current, selected)
        
        // Try selecting current interval if it doesn't overlap
        if len(selected) == 0 || intervals[index][0] >= selected[len(selected)-1][1] {
            backtrack(index+1, current, append(selected, intervals[index]))
        }
    }
    
    backtrack(0, []int{}, [][]int{})
    return result
}
```

#### **Return Statistics**
```go
import "sort"

type IntervalStats struct {
    TotalIntervals     int
    NonOverlapping     int
    Removed            int
    MaxOverlap         int
    OverlappingPairs   [][]int
}

func intervalStatistics(intervals [][]int) IntervalStats {
    if len(intervals) == 0 {
        return IntervalStats{}
    }
    
    // Sort by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    nonOverlapping := 1
    end := intervals[0][1]
    maxOverlap := 0
    var overlappingPairs [][]int
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] >= end {
            nonOverlapping++
            end = intervals[i][1]
        } else {
            overlappingPairs = append(overlappingPairs, []int{i-1, i})
            maxOverlap = max(maxOverlap, 2)
        }
    }
    
    return IntervalStats{
        TotalIntervals:   len(intervals),
        NonOverlapping:   nonOverlapping,
        Removed:          len(intervals) - nonOverlapping,
        MaxOverlap:       maxOverlap,
        OverlappingPairs: overlappingPairs,
    }
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Complexity
- **Time Complexity:** O(n log n) for sorting
- **Space Complexity:** O(1) for in-place, O(n) for additional arrays