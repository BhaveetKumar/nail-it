---
# Auto-generated front matter
Title: Slidingwindowmaximum
LastUpdated: 2025-11-06T20:45:58.711753
Tags: []
Status: draft
---

# Sliding Window Maximum

### Problem
You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

**Example:**
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

### Golang Solution

```go
func maxSlidingWindow(nums []int, k int) []int {
    if len(nums) == 0 || k == 0 {
        return []int{}
    }
    
    var result []int
    var deque []int // Store indices
    
    for i := 0; i < len(nums); i++ {
        // Remove indices outside current window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        
        // Remove indices of elements smaller than current element
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        
        // Add current index
        deque = append(deque, i)
        
        // Add maximum to result when window is complete
        if i >= k-1 {
            result = append(result, nums[deque[0]])
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Heap**
```go
import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] > h[j] }
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

func maxSlidingWindowHeap(nums []int, k int) []int {
    if len(nums) == 0 || k == 0 {
        return []int{}
    }
    
    var result []int
    h := &IntHeap{}
    heap.Init(h)
    
    // Initialize first window
    for i := 0; i < k; i++ {
        heap.Push(h, nums[i])
    }
    
    result = append(result, (*h)[0])
    
    // Slide the window
    for i := k; i < len(nums); i++ {
        // Remove leftmost element (this is simplified - in practice you'd need to track indices)
        heap.Push(h, nums[i])
        result = append(result, (*h)[0])
    }
    
    return result
}
```

#### **Using Two Pointers**
```go
func maxSlidingWindowTwoPointers(nums []int, k int) []int {
    if len(nums) == 0 || k == 0 {
        return []int{}
    }
    
    var result []int
    
    for i := 0; i <= len(nums)-k; i++ {
        max := nums[i]
        for j := i; j < i+k; j++ {
            if nums[j] > max {
                max = nums[j]
            }
        }
        result = append(result, max)
    }
    
    return result
}
```

#### **Return with Indices**
```go
type WindowResult struct {
    Max   int
    Start int
    End   int
}

func maxSlidingWindowWithIndices(nums []int, k int) []WindowResult {
    if len(nums) == 0 || k == 0 {
        return []WindowResult{}
    }
    
    var result []WindowResult
    var deque []int // Store indices
    
    for i := 0; i < len(nums); i++ {
        // Remove indices outside current window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        
        // Remove indices of elements smaller than current element
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        
        // Add current index
        deque = append(deque, i)
        
        // Add maximum to result when window is complete
        if i >= k-1 {
            result = append(result, WindowResult{
                Max:   nums[deque[0]],
                Start: i - k + 1,
                End:   i,
            })
        }
    }
    
    return result
}
```

#### **Return All Elements in Each Window**
```go
func slidingWindowElements(nums []int, k int) [][]int {
    if len(nums) == 0 || k == 0 {
        return [][]int{}
    }
    
    var result [][]int
    
    for i := 0; i <= len(nums)-k; i++ {
        window := make([]int, k)
        copy(window, nums[i:i+k])
        result = append(result, window)
    }
    
    return result
}
```

#### **Return Statistics for Each Window**
```go
type WindowStats struct {
    Max   int
    Min   int
    Sum   int
    Avg   float64
    Count int
}

func slidingWindowStats(nums []int, k int) []WindowStats {
    if len(nums) == 0 || k == 0 {
        return []WindowStats{}
    }
    
    var result []WindowStats
    
    for i := 0; i <= len(nums)-k; i++ {
        max := nums[i]
        min := nums[i]
        sum := 0
        
        for j := i; j < i+k; j++ {
            if nums[j] > max {
                max = nums[j]
            }
            if nums[j] < min {
                min = nums[j]
            }
            sum += nums[j]
        }
        
        result = append(result, WindowStats{
            Max:   max,
            Min:   min,
            Sum:   sum,
            Avg:   float64(sum) / float64(k),
            Count: k,
        })
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) for deque approach, O(n×k) for two pointers
- **Space Complexity:** O(k) for deque, O(1) for two pointers
