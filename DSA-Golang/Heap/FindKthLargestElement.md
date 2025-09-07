# Find Kth Largest Element in an Array

### Problem
Given an integer array `nums` and an integer `k`, return the kth largest element in the array.

Note that it is the kth largest element in sorted order, not the kth distinct element.

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
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] } // Min heap
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
    minHeap := &IntHeap{}
    heap.Init(minHeap)
    
    for _, num := range nums {
        heap.Push(minHeap, num)
        if minHeap.Len() > k {
            heap.Pop(minHeap)
        }
    }
    
    return (*minHeap)[0]
}
```

### Alternative Solutions

#### **Quick Select Algorithm**
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

#### **Sorting Approach**
```go
import "sort"

func findKthLargestSort(nums []int, k int) int {
    sort.Ints(nums)
    return nums[len(nums)-k]
}
```

### Complexity
- **Time Complexity:** O(n log k) for heap, O(n) average for quick select, O(n log n) for sorting
- **Space Complexity:** O(k) for heap, O(1) for quick select
