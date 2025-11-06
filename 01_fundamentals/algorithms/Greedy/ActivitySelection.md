---
# Auto-generated front matter
Title: Activityselection
LastUpdated: 2025-11-06T20:45:58.728895
Tags: []
Status: draft
---

# Activity Selection Problem

### Problem
You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.

**Example:**
```
Input: activities = [[1,3],[2,5],[0,6],[5,7],[8,9],[5,9]]
Output: 4
Explanation: The maximum number of activities is 4: [1,3], [5,7], [8,9], [5,9]
```

### Golang Solution

```go
import "sort"

type Activity struct {
    start, finish int
}

func activitySelection(activities [][]int) int {
    if len(activities) == 0 {
        return 0
    }
    
    // Convert to Activity structs and sort by finish time
    acts := make([]Activity, len(activities))
    for i, activity := range activities {
        acts[i] = Activity{start: activity[0], finish: activity[1]}
    }
    
    sort.Slice(acts, func(i, j int) bool {
        return acts[i].finish < acts[j].finish
    })
    
    count := 1
    lastFinish := acts[0].finish
    
    for i := 1; i < len(acts); i++ {
        if acts[i].start >= lastFinish {
            count++
            lastFinish = acts[i].finish
        }
    }
    
    return count
}
```

### Alternative Solutions

#### **With Selected Activities**
```go
func activitySelectionWithDetails(activities [][]int) []int {
    if len(activities) == 0 {
        return []int{}
    }
    
    // Create indices and sort by finish time
    indices := make([]int, len(activities))
    for i := range indices {
        indices[i] = i
    }
    
    sort.Slice(indices, func(i, j int) bool {
        return activities[indices[i]][1] < activities[indices[j]][1]
    })
    
    var selected []int
    selected = append(selected, indices[0])
    lastFinish := activities[indices[0]][1]
    
    for i := 1; i < len(indices); i++ {
        idx := indices[i]
        if activities[idx][0] >= lastFinish {
            selected = append(selected, idx)
            lastFinish = activities[idx][1]
        }
    }
    
    return selected
}
```

### Complexity
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(n)
