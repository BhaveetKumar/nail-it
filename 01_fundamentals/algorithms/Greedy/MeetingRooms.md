---
# Auto-generated front matter
Title: Meetingrooms
LastUpdated: 2025-11-06T20:45:58.728769
Tags: []
Status: draft
---

# Meeting Rooms

### Problem
Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, determine if a person could attend all meetings.

**Example:**
```
Input: intervals = [[0,30],[5,10],[15,20]]
Output: false

Input: intervals = [[7,10],[2,4]]
Output: true
```

### Golang Solution

```go
import "sort"

func canAttendMeetings(intervals [][]int) bool {
    if len(intervals) <= 1 {
        return true
    }
    
    // Sort intervals by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    // Check for overlaps
    for i := 1; i < len(intervals); i++ {
        if intervals[i][0] < intervals[i-1][1] {
            return false
        }
    }
    
    return true
}
```

### Alternative Solutions

#### **Meeting Rooms II - Minimum Rooms Needed**
```go
func minMeetingRooms(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    // Separate start and end times
    starts := make([]int, len(intervals))
    ends := make([]int, len(intervals))
    
    for i, interval := range intervals {
        starts[i] = interval[0]
        ends[i] = interval[1]
    }
    
    sort.Ints(starts)
    sort.Ints(ends)
    
    rooms := 0
    endIndex := 0
    
    for i := 0; i < len(starts); i++ {
        if starts[i] < ends[endIndex] {
            rooms++
        } else {
            endIndex++
        }
    }
    
    return rooms
}
```

#### **Using Priority Queue**
```go
import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
    *h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func minMeetingRoomsPQ(intervals [][]int) int {
    if len(intervals) == 0 {
        return 0
    }
    
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i][0] < intervals[j][0]
    })
    
    minHeap := &IntHeap{}
    heap.Init(minHeap)
    
    for _, interval := range intervals {
        if minHeap.Len() > 0 && (*minHeap)[0] <= interval[0] {
            heap.Pop(minHeap)
        }
        heap.Push(minHeap, interval[1])
    }
    
    return minHeap.Len()
}
```

### Complexity
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1) for basic, O(n) for PQ approach
