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
    
    count := 0
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] < end {
            count++
        } else {
            end = intervals[i][1]
        }
    }
    
    return count
}
```

### Alternative Solutions

#### **Sort by Start Time**
```go
func eraseOverlapIntervalsStart(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    // Sort by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    count := 0
    end := intervals[0][1]
    
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] < end {
            count++
            // Keep the interval with smaller end time
            if intervals[i][1] < end {
                end = intervals[i][1]
            }
        } else {
            end = intervals[i][1]
        }
    }
    
    return count
}
```

#### **Using Priority Queue**
```go
import "container/heap"

type IntervalHeap [][]int

func (h IntervalHeap) Len() int           { return len(h) }
func (h IntervalHeap) Less(i, j int) bool { return h[i][1] < h[j][1] }
func (h IntervalHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntervalHeap) Push(x interface{}) {
    *h = append(*h, x.([]int))
}

func (h *IntervalHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func eraseOverlapIntervalsPQ(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    h := &IntervalHeap{}
    heap.Init(h)
    
    for _, interval := range intervals {
        heap.Push(h, interval)
    }
    
    count := 0
    end := heap.Pop(h).([]int)[1]
    
    for h.Len() > 0 {
        current := heap.Pop(h).([]int)
        if current[0] < end {
            count++
        } else {
            end = current[1]
        }
    }
    
    return count
}
```

### Complexity
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)
