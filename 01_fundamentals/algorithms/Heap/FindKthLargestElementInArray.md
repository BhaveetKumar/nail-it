---
# Auto-generated front matter
Title: Findkthlargestelementinarray
LastUpdated: 2025-11-06T20:45:58.719192
Tags: []
Status: draft
---

# Find Kth Largest Element in an Array

### Problem
Given an integer array `nums` and an integer `k`, return the `kth` largest element in the array.

Note that it is the `kth` largest element in sorted order, not the `kth` distinct element.

**Example:**
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

### Golang Solution

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

func findKthLargest(nums []int, k int) int {
    h := &IntHeap{}
    heap.Init(h)
    
    for _, num := range nums {
        heap.Push(h, num)
        if h.Len() > k {
            heap.Pop(h)
        }
    }
    
    return (*h)[0]
}
```

### Alternative Solutions

#### **Using Max Heap**
```go
type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
    *h = append(*h, x.(int))
}

func (h *MaxHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func findKthLargestMaxHeap(nums []int, k int) int {
    h := &MaxHeap{}
    heap.Init(h)
    
    for _, num := range nums {
        heap.Push(h, num)
    }
    
    for i := 0; i < k-1; i++ {
        heap.Pop(h)
    }
    
    return heap.Pop(h).(int)
}
```

#### **Using Quick Select**
```go
func findKthLargestQuickSelect(nums []int, k int) int {
    return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, left, right, k int) int {
    if left == right {
        return nums[left]
    }
    
    pivotIndex := partition(nums, left, right)
    
    if k == pivotIndex {
        return nums[k]
    } else if k < pivotIndex {
        return quickSelect(nums, left, pivotIndex-1, k)
    } else {
        return quickSelect(nums, pivotIndex+1, right, k)
    }
}

func partition(nums []int, left, right int) int {
    pivot := nums[right]
    i := left
    
    for j := left; j < right; j++ {
        if nums[j] <= pivot {
            nums[i], nums[j] = nums[j], nums[i]
            i++
        }
    }
    
    nums[i], nums[right] = nums[right], nums[i]
    return i
}
```

#### **Using Sorting**
```go
import "sort"

func findKthLargestSort(nums []int, k int) int {
    sort.Ints(nums)
    return nums[len(nums)-k]
}
```

#### **Using Counting Sort**
```go
func findKthLargestCountingSort(nums []int, k int) int {
    minVal, maxVal := nums[0], nums[0]
    
    for _, num := range nums {
        if num < minVal {
            minVal = num
        }
        if num > maxVal {
            maxVal = num
        }
    }
    
    count := make([]int, maxVal-minVal+1)
    
    for _, num := range nums {
        count[num-minVal]++
    }
    
    for i := len(count) - 1; i >= 0; i-- {
        k -= count[i]
        if k <= 0 {
            return i + minVal
        }
    }
    
    return -1
}
```

#### **Return Top K Elements**
```go
func findTopKLargest(nums []int, k int) []int {
    h := &IntHeap{}
    heap.Init(h)
    
    for _, num := range nums {
        heap.Push(h, num)
        if h.Len() > k {
            heap.Pop(h)
        }
    }
    
    result := make([]int, k)
    for i := k - 1; i >= 0; i-- {
        result[i] = heap.Pop(h).(int)
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n log k) for heap, O(n) average for quick select, O(n log n) for sorting
- **Space Complexity:** O(k) for heap, O(1) for quick select
