# Heap Sort

### Problem
Implement Heap Sort algorithm to sort an array of integers.

Heap Sort is a comparison-based sorting algorithm that uses a binary heap data structure.

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
        nums[0], nums[i] = nums[i], nums[0]
        heapify(nums, i, 0)
    }
}

func heapify(nums []int, n, i int) {
    largest := i
    left := 2*i + 1
    right := 2*i + 2
    
    if left < n && nums[left] > nums[largest] {
        largest = left
    }
    
    if right < n && nums[right] > nums[largest] {
        largest = right
    }
    
    if largest != i {
        nums[i], nums[largest] = nums[largest], nums[i]
        heapify(nums, n, largest)
    }
}
```

### Alternative Solutions

#### **Using Min Heap**
```go
func heapSortMinHeap(nums []int) {
    n := len(nums)
    
    // Build min heap
    for i := n/2 - 1; i >= 0; i-- {
        heapifyMin(nums, n, i)
    }
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        heapifyMin(nums, i, 0)
    }
}

func heapifyMin(nums []int, n, i int) {
    smallest := i
    left := 2*i + 1
    right := 2*i + 2
    
    if left < n && nums[left] < nums[smallest] {
        smallest = left
    }
    
    if right < n && nums[right] < nums[smallest] {
        smallest = right
    }
    
    if smallest != i {
        nums[i], nums[smallest] = nums[smallest], nums[i]
        heapifyMin(nums, n, smallest)
    }
}
```

#### **Using Container/Heap**
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

func heapSortContainer(nums []int) []int {
    h := &IntHeap{}
    heap.Init(h)
    
    // Push all elements to heap
    for _, num := range nums {
        heap.Push(h, num)
    }
    
    // Pop elements to get sorted array
    var result []int
    for h.Len() > 0 {
        result = append(result, heap.Pop(h).(int))
    }
    
    return result
}
```

#### **With Counters**
```go
func heapSortWithCounters(nums []int) ([]int, int, int) {
    comparisons := 0
    swaps := 0
    n := len(nums)
    
    // Build max heap
    for i := n/2 - 1; i >= 0; i-- {
        heapifyWithCounters(nums, n, i, &comparisons, &swaps)
    }
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        swaps++
        heapifyWithCounters(nums, i, 0, &comparisons, &swaps)
    }
    
    return nums, comparisons, swaps
}

func heapifyWithCounters(nums []int, n, i int, comparisons, swaps *int) {
    largest := i
    left := 2*i + 1
    right := 2*i + 2
    
    if left < n {
        (*comparisons)++
        if nums[left] > nums[largest] {
            largest = left
        }
    }
    
    if right < n {
        (*comparisons)++
        if nums[right] > nums[largest] {
            largest = right
        }
    }
    
    if largest != i {
        nums[i], nums[largest] = nums[largest], nums[i]
        (*swaps)++
        heapifyWithCounters(nums, n, largest, comparisons, swaps)
    }
}
```

#### **Return Heap Structure**
```go
type HeapNode struct {
    Value int
    Left  *HeapNode
    Right *HeapNode
}

func heapSortWithStructure(nums []int) ([]int, *HeapNode) {
    n := len(nums)
    
    // Build max heap
    for i := n/2 - 1; i >= 0; i-- {
        heapify(nums, n, i)
    }
    
    // Create heap structure
    root := buildHeapStructure(nums, 0)
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        heapify(nums, i, 0)
    }
    
    return nums, root
}

func buildHeapStructure(nums []int, index int) *HeapNode {
    if index >= len(nums) {
        return nil
    }
    
    node := &HeapNode{Value: nums[index]}
    left := 2*index + 1
    right := 2*index + 2
    
    if left < len(nums) {
        node.Left = buildHeapStructure(nums, left)
    }
    
    if right < len(nums) {
        node.Right = buildHeapStructure(nums, right)
    }
    
    return node
}
```

#### **Return All Steps**
```go
type HeapSortStep struct {
    Array      []int
    Step       string
    Index      int
    Comparison []int
}

func heapSortWithSteps(nums []int) []HeapSortStep {
    var steps []HeapSortStep
    n := len(nums)
    
    // Build max heap
    for i := n/2 - 1; i >= 0; i-- {
        steps = append(steps, HeapSortStep{
            Array: append([]int{}, nums...),
            Step:  "Building heap",
            Index: i,
        })
        heapify(nums, n, i)
    }
    
    // Extract elements from heap one by one
    for i := n - 1; i > 0; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        steps = append(steps, HeapSortStep{
            Array: append([]int{}, nums...),
            Step:  "Swapping",
            Index: i,
        })
        heapify(nums, i, 0)
    }
    
    return steps
}
```

### Complexity
- **Time Complexity:** O(n log n) in all cases
- **Space Complexity:** O(1) for in-place, O(n) for additional structures