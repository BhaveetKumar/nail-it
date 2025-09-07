# Heap Pattern

> **Master heap data structures and priority queue algorithms with Go implementations**

## 📋 Problems

### **Basic Heap Operations**
- [Kth Largest Element in an Array](./KthLargestElementInArray.md) - Find Kth largest element
- [Kth Smallest Element in a BST](./KthSmallestElementInBST.md) - Find Kth smallest in BST
- [Find Median from Data Stream](./FindMedianFromDataStream.md) - Maintain median of stream
- [Merge k Sorted Lists](./MergeKSortedLists.md) - Merge multiple sorted lists
- [Top K Frequent Elements](./TopKFrequentElements.md) - Find most frequent elements

### **Heap Applications**
- [Last Stone Weight](./LastStoneWeight.md) - Simulate stone crushing
- [Minimum Cost to Connect Sticks](./MinimumCostToConnectSticks.md) - Connect sticks optimally
- [Reorganize String](./ReorganizeString.md) - Rearrange string characters
- [Task Scheduler](./TaskScheduler.md) - Schedule tasks with cooldown
- [Maximum Performance of a Team](./MaximumPerformanceOfTeam.md) - Optimize team performance

### **Advanced Heap Problems**
- [Sliding Window Median](./SlidingWindowMedian.md) - Find median in sliding window
- [IPO](./IPO.md) - Initial Public Offering optimization
- [Course Schedule III](./CourseScheduleIII.md) - Maximum courses to take
- [Maximum Number of Events](./MaximumNumberOfEvents.md) - Attend maximum events
- [Furthest Building You Can Reach](./FurthestBuildingYouCanReach.md) - Reach furthest building

---

## 🎯 Key Concepts

### **Heap Properties**
- **Min Heap**: Parent ≤ children (root is minimum)
- **Max Heap**: Parent ≥ children (root is maximum)
- **Complete Binary Tree**: All levels filled except possibly last
- **Heap Property**: Maintained after insert/delete operations

### **Heap Operations**
- **Insert**: O(log n) - Add element and heapify up
- **Delete**: O(log n) - Remove root and heapify down
- **Peek**: O(1) - View root element
- **Build Heap**: O(n) - Build heap from array

### **Common Patterns**
- **Top K Elements**: Use min heap of size K
- **Stream Processing**: Use two heaps for median
- **Merging**: Use heap for merging sorted sequences
- **Scheduling**: Use heap for task scheduling

---

## 🛠️ Go-Specific Tips

### **Go's Built-in Heap**
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

func main() {
    h := &IntHeap{2, 1, 5}
    heap.Init(h)
    heap.Push(h, 3)
    fmt.Printf("minimum: %d\n", (*h)[0])
    for h.Len() > 0 {
        fmt.Printf("%d ", heap.Pop(h))
    }
}
```

### **Kth Largest Element**
```go
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

### **Two Heaps for Median**
```go
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
    if mf.maxHeap.Len() == 0 || num <= (*mf.maxHeap)[0] {
        heap.Push(mf.maxHeap, num)
    } else {
        heap.Push(mf.minHeap, num)
    }
    
    // Balance heaps
    if mf.maxHeap.Len() > mf.minHeap.Len()+1 {
        heap.Push(mf.minHeap, heap.Pop(mf.maxHeap))
    } else if mf.minHeap.Len() > mf.maxHeap.Len()+1 {
        heap.Push(mf.maxHeap, heap.Pop(mf.minHeap))
    }
}

func (mf *MedianFinder) FindMedian() float64 {
    if mf.maxHeap.Len() > mf.minHeap.Len() {
        return float64((*mf.maxHeap)[0])
    } else if mf.minHeap.Len() > mf.maxHeap.Len() {
        return float64((*mf.minHeap)[0])
    } else {
        return float64((*mf.maxHeap)[0]+(*mf.minHeap)[0]) / 2.0
    }
}
```

### **Custom Heap for Complex Data**
```go
type Item struct {
    value    int
    priority int
    index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    return pq[i].priority > pq[j].priority // Max heap
}

func (pq PriorityQueue) Swap(i, j int) {
    pq[i], pq[j] = pq[j], pq[i]
    pq[i].index = i
    pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
    n := len(*pq)
    item := x.(*Item)
    item.index = n
    *pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    old[n-1] = nil
    item.index = -1
    *pq = old[0 : n-1]
    return item
}
```

---

## 🎯 Interview Tips

### **How to Identify Heap Problems**
1. **Top K Elements**: Find K largest/smallest elements
2. **Stream Processing**: Process data streams with heaps
3. **Merging**: Merge multiple sorted sequences
4. **Scheduling**: Optimize task scheduling
5. **Median**: Find median in data stream

### **Common Heap Problem Patterns**
- **Top K**: Use min heap of size K for largest elements
- **Two Heaps**: Use two heaps for median finding
- **Merging**: Use heap for merging sorted sequences
- **Scheduling**: Use heap for task scheduling
- **Optimization**: Use heap for optimization problems

### **Optimization Tips**
- **Use Built-in Heap**: Leverage Go's container/heap package
- **Custom Comparators**: Define custom comparison functions
- **Heap Size**: Limit heap size for memory efficiency
- **Early Termination**: Stop when solution is found
