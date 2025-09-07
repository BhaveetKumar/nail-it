# Heap Sort

### Problem
Implement Heap Sort algorithm to sort an array of integers.

Heap Sort is a comparison-based sorting algorithm that uses a binary heap data structure. It has O(n log n) time complexity and is an in-place algorithm.

**Example:**
```
Input: [64, 34, 25, 12, 22, 11, 90]
Output: [11, 12, 22, 25, 34, 64, 90]
```

### Golang Solution

```go
func heapSort(nums []int) {
    n := len(nums)
    
    // Build max heap
    for i := n/2 - 1; i >= 0; i-- {
        heapify(nums, n, i)
    }
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        // Move current root to end
        nums[0], nums[i] = nums[i], nums[0]
        
        // Call heapify on the reduced heap
        heapify(nums, i, 0)
    }
}

func heapify(nums []int, n, i int) {
    largest := i
    left := 2*i + 1
    right := 2*i + 2
    
    // If left child is larger than root
    if left < n && nums[left] > nums[largest] {
        largest = left
    }
    
    // If right child is larger than largest so far
    if right < n && nums[right] > nums[largest] {
        largest = right
    }
    
    // If largest is not root
    if largest != i {
        nums[i], nums[largest] = nums[largest], nums[i]
        
        // Recursively heapify the affected sub-tree
        heapify(nums, n, largest)
    }
}
```

### Alternative Solutions

#### **Using Go's container/heap**
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

func heapSortWithContainer(nums []int) {
    h := &IntHeap{}
    heap.Init(h)
    
    // Push all elements to heap
    for _, num := range nums {
        heap.Push(h, num)
    }
    
    // Pop elements to get sorted order
    for i := 0; i < len(nums); i++ {
        nums[i] = heap.Pop(h).(int)
    }
}
```

### Complexity
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)
