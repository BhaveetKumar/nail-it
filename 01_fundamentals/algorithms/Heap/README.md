---
# Auto-generated front matter
Title: Readme
LastUpdated: 2025-11-06T20:45:58.717928
Tags: []
Status: draft
---

# Heap Pattern

> **Master heap data structures and priority queue algorithms with Go implementations**

## 📋 Problems

### **Basic Heap Operations**

- [Kth Largest Element in an Array](KthLargestElementInArray.md) - Find Kth largest element
- [Kth Smallest Element in a BST](../Trees/KthSmallestElementInBST.md) - Find Kth smallest in BST
- [Find Median from Data Stream](FindMedianFromDataStream.md) - Maintain median of stream
- [Merge k Sorted Lists](MergeKSortedLists.md) - Merge multiple sorted lists
- [Top K Frequent Elements](TopKFrequentElements.md) - Find most frequent elements

### **Heap Applications**

- [Last Stone Weight](LastStoneWeight.md) - Simulate stone crushing
- [Minimum Cost to Connect Sticks](MinimumCostToConnectSticks.md) - Connect sticks optimally
- [Reorganize String](ReorganizeString.md) - Rearrange string characters
- [Task Scheduler](TaskScheduler.md) - Schedule tasks with cooldown
- [Maximum Performance of a Team](MaximumPerformanceOfTeam.md) - Optimize team performance

### **Advanced Heap Problems**

- [Sliding Window Median](SlidingWindowMedian.md) - Find median in sliding window
- [IPO](IPO.md) - Initial Public Offering optimization
- [Course Schedule III](CourseScheduleIII.md) - Maximum courses to take
- [Maximum Number of Events](MaximumNumberOfEvents.md) - Attend maximum events
- [Furthest Building You Can Reach](FurthestBuildingYouCanReach.md) - Reach furthest building

---

## 🎯 Key Concepts

### **Heap Properties**

**Detailed Explanation:**
A heap is a specialized tree-based data structure that satisfies the heap property. It's a complete binary tree where each parent node has a specific relationship with its children, making it highly efficient for priority-based operations.

**Min Heap Properties:**

- **Heap Property**: Parent ≤ children for all nodes
- **Root Element**: Always contains the minimum value
- **Complete Binary Tree**: All levels are completely filled except possibly the last level
- **Left-to-Right Filling**: Last level is filled from left to right
- **Height**: O(log n) where n is the number of elements

**Max Heap Properties:**

- **Heap Property**: Parent ≥ children for all nodes
- **Root Element**: Always contains the maximum value
- **Complete Binary Tree**: Same structure as min heap
- **Use Cases**: Finding maximum elements, priority queues with high priority first

**Why Heaps Are Efficient:**

- **O(log n) Operations**: Insert and delete operations are logarithmic
- **O(1) Access**: Root element access is constant time
- **Memory Efficient**: Can be implemented using arrays
- **Cache Friendly**: Array-based implementation provides good cache locality

**Mathematical Properties:**

```
Height of heap: ⌊log₂(n)⌋
Number of leaves: ⌈n/2⌉
Number of internal nodes: ⌊n/2⌋
Parent of node i: ⌊(i-1)/2⌋
Left child of node i: 2i + 1
Right child of node i: 2i + 2
```

### **Heap Operations**

**Detailed Explanation:**
Heap operations are designed to maintain the heap property while providing efficient access to the most important element (minimum or maximum).

**Insert Operation (Heapify Up):**

- **Process**: Add element to the end and bubble up to maintain heap property
- **Time Complexity**: O(log n)
- **Steps**:
  1. Add element to the last position
  2. Compare with parent and swap if necessary
  3. Continue until heap property is satisfied
- **Use Case**: Adding new elements to priority queue

**Delete Operation (Heapify Down):**

- **Process**: Remove root and replace with last element, then bubble down
- **Time Complexity**: O(log n)
- **Steps**:
  1. Remove root element
  2. Move last element to root position
  3. Compare with children and swap with smaller/larger child
  4. Continue until heap property is satisfied
- **Use Case**: Removing highest priority element

**Peek Operation:**

- **Process**: Return root element without removing it
- **Time Complexity**: O(1)
- **Use Case**: Checking the next element to process

**Build Heap Operation:**

- **Process**: Convert an array into a heap
- **Time Complexity**: O(n) - more efficient than n insertions
- **Algorithm**: Start from the last internal node and heapify down
- **Use Case**: Initializing heap from existing data

**Heapify Algorithms:**

```go
// Heapify Up - used in insert
func heapifyUp(heap []int, index int) {
    for index > 0 {
        parent := (index - 1) / 2
        if heap[index] >= heap[parent] {
            break
        }
        heap[index], heap[parent] = heap[parent], heap[index]
        index = parent
    }
}

// Heapify Down - used in delete and build heap
func heapifyDown(heap []int, index, size int) {
    for {
        smallest := index
        left := 2*index + 1
        right := 2*index + 2

        if left < size && heap[left] < heap[smallest] {
            smallest = left
        }
        if right < size && heap[right] < heap[smallest] {
            smallest = right
        }

        if smallest == index {
            break
        }

        heap[index], heap[smallest] = heap[smallest], heap[index]
        index = smallest
    }
}
```

### **Common Patterns**

**Detailed Explanation:**
Understanding common heap patterns is crucial for solving heap-related problems efficiently. Each pattern has specific use cases and implementation strategies.

**Top K Elements Pattern:**

- **Use Case**: Find K largest or smallest elements
- **Strategy**: Use min heap of size K for largest elements, max heap for smallest
- **Why It Works**: Maintains only the K most important elements
- **Time Complexity**: O(n log k) instead of O(n log n)
- **Space Complexity**: O(k)
- **Example**: Find top 10 most frequent words in a document

**Stream Processing Pattern:**

- **Use Case**: Process data streams and maintain running statistics
- **Strategy**: Use two heaps to maintain median or other statistics
- **Why It Works**: Allows efficient updates as new data arrives
- **Time Complexity**: O(log n) per element
- **Space Complexity**: O(n)
- **Example**: Find median of a stream of numbers

**Merging Pattern:**

- **Use Case**: Merge multiple sorted sequences
- **Strategy**: Use heap to always process the smallest element
- **Why It Works**: Heap provides efficient access to minimum element
- **Time Complexity**: O(n log k) where k is number of sequences
- **Space Complexity**: O(k)
- **Example**: Merge K sorted linked lists

**Scheduling Pattern:**

- **Use Case**: Optimize task scheduling and resource allocation
- **Strategy**: Use heap to prioritize tasks based on criteria
- **Why It Works**: Heap maintains tasks in priority order
- **Time Complexity**: O(log n) per task
- **Space Complexity**: O(n)
- **Example**: CPU task scheduling with priority

**Discussion Questions & Answers:**

**Q1: How do you choose between min heap and max heap for different problems?**

**Answer:** Heap selection strategy:

- **Min Heap**: Use when you need to find minimum elements or maintain smallest K elements
- **Max Heap**: Use when you need to find maximum elements or maintain largest K elements
- **Two Heaps**: Use both for median finding or range queries
- **Custom Heap**: Use when you need custom ordering (e.g., by frequency, by deadline)
- **Considerations**:
  - Problem requirements (min vs max)
  - Memory constraints (heap size)
  - Performance requirements (access patterns)
  - Data characteristics (sorted vs unsorted)

**Q2: What are the trade-offs between using Go's built-in heap vs implementing a custom heap?**

**Answer:** Implementation trade-offs:

- **Built-in Heap (container/heap)**:
  - **Advantages**: Well-tested, efficient, follows Go conventions
  - **Disadvantages**: Less control, generic interface, potential overhead
  - **Best for**: Standard heap operations, learning, prototyping
- **Custom Heap**:
  - **Advantages**: Optimized for specific use case, better performance, more control
  - **Disadvantages**: More code to maintain, potential bugs, reinventing the wheel
  - **Best for**: Performance-critical applications, specific requirements
- **Hybrid Approach**: Use built-in for development, custom for production optimization

**Q3: How do you optimize heap operations for large datasets?**

**Answer:** Optimization strategies:

- **Memory Management**: Use object pooling to reduce garbage collection
- **Batch Operations**: Process multiple elements at once when possible
- **Lazy Evaluation**: Defer expensive operations until necessary
- **Custom Comparators**: Optimize comparison functions for specific data types
- **Heap Size Limits**: Limit heap size to prevent memory issues
- **Parallel Processing**: Use multiple heaps for parallel processing
- **Data Structures**: Consider alternative data structures for specific use cases
- **Caching**: Cache frequently accessed elements

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
