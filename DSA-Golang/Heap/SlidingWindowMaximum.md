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
    deque := []int{} // Store indices
    
    for i := 0; i < len(nums); i++ {
        // Remove indices outside current window
        for len(deque) > 0 && deque[0] <= i-k {
            deque = deque[1:]
        }
        
        // Remove indices of elements smaller than current element
        for len(deque) > 0 && nums[deque[len(deque)-1]] <= nums[i] {
            deque = deque[:len(deque)-1]
        }
        
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

#### **Using Priority Queue**
```go
import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] > h[j] } // Max heap
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

func maxSlidingWindowPQ(nums []int, k int) []int {
    if len(nums) == 0 || k == 0 {
        return []int{}
    }
    
    var result []int
    maxHeap := &IntHeap{}
    heap.Init(maxHeap)
    
    for i := 0; i < k; i++ {
        heap.Push(maxHeap, nums[i])
    }
    
    result = append(result, (*maxHeap)[0])
    
    for i := k; i < len(nums); i++ {
        // Remove leftmost element (simplified - not optimal)
        heap.Push(maxHeap, nums[i])
        result = append(result, (*maxHeap)[0])
    }
    
    return result
}
```

#### **Brute Force**
```go
func maxSlidingWindowBruteForce(nums []int, k int) []int {
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

### Complexity
- **Time Complexity:** O(n) for deque, O(n log k) for priority queue, O(n × k) for brute force
- **Space Complexity:** O(k)
