# Find Median from Data Stream

### Problem
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

Implement the MedianFinder class:
- `MedianFinder()` initializes the MedianFinder object.
- `void addNum(int num)` adds the integer num from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far.

**Example:**
```
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]

Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
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

type MedianFinder struct {
    maxHeap *IntHeap // Left half (max heap)
    minHeap *IntHeap // Right half (min heap)
}

func Constructor() MedianFinder {
    return MedianFinder{
        maxHeap: &IntHeap{},
        minHeap: &IntHeap{},
    }
}

func (mf *MedianFinder) AddNum(num int) {
    // Add to max heap first
    heap.Push(mf.maxHeap, -num) // Use negative for max heap behavior
    
    // Balance the heaps
    heap.Push(mf.minHeap, -heap.Pop(mf.maxHeap).(int))
    
    // Ensure max heap is not smaller than min heap
    if mf.maxHeap.Len() < mf.minHeap.Len() {
        heap.Push(mf.maxHeap, -heap.Pop(mf.minHeap).(int))
    }
}

func (mf *MedianFinder) FindMedian() float64 {
    if mf.maxHeap.Len() > mf.minHeap.Len() {
        return float64(-(*mf.maxHeap)[0])
    }
    return float64(-(*mf.maxHeap)[0]+(*mf.minHeap)[0]) / 2.0
}
```

### Alternative Solutions

#### **Using Two Heaps with Proper Max Heap**
```go
type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] } // Max heap
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

type MedianFinderTwoHeaps struct {
    maxHeap *MaxHeap // Left half
    minHeap *IntHeap // Right half
}

func NewMedianFinder() *MedianFinderTwoHeaps {
    return &MedianFinderTwoHeaps{
        maxHeap: &MaxHeap{},
        minHeap: &IntHeap{},
    }
}

func (mf *MedianFinderTwoHeaps) AddNum(num int) {
    heap.Push(mf.maxHeap, num)
    
    heap.Push(mf.minHeap, heap.Pop(mf.maxHeap).(int))
    
    if mf.maxHeap.Len() < mf.minHeap.Len() {
        heap.Push(mf.maxHeap, heap.Pop(mf.minHeap).(int))
    }
}

func (mf *MedianFinderTwoHeaps) FindMedian() float64 {
    if mf.maxHeap.Len() > mf.minHeap.Len() {
        return float64((*mf.maxHeap)[0])
    }
    return float64((*mf.maxHeap)[0]+(*mf.minHeap)[0]) / 2.0
}
```

### Complexity
- **Time Complexity:** O(log n) for addNum, O(1) for findMedian
- **Space Complexity:** O(n)
