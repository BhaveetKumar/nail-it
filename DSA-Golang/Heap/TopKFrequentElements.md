# Top K Frequent Elements

### Problem
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

**Example:**
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Input: nums = [1], k = 1
Output: [1]
```

### Golang Solution

```go
import "container/heap"

type Element struct {
    value int
    count int
}

type MaxHeap []Element

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].count > h[j].count }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
    *h = append(*h, x.(Element))
}

func (h *MaxHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func topKFrequent(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    h := &MaxHeap{}
    heap.Init(h)
    
    for value, freq := range count {
        heap.Push(h, Element{value, freq})
    }
    
    result := make([]int, k)
    for i := 0; i < k; i++ {
        element := heap.Pop(h).(Element)
        result[i] = element.value
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Min Heap**
```go
type MinHeap []Element

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].count < h[j].count }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
    *h = append(*h, x.(Element))
}

func (h *MinHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func topKFrequentMinHeap(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    h := &MinHeap{}
    heap.Init(h)
    
    for value, freq := range count {
        heap.Push(h, Element{value, freq})
        
        if h.Len() > k {
            heap.Pop(h)
        }
    }
    
    result := make([]int, k)
    for i := k - 1; i >= 0; i-- {
        element := heap.Pop(h).(Element)
        result[i] = element.value
    }
    
    return result
}
```

#### **Using Bucket Sort**
```go
func topKFrequentBucketSort(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    // Create buckets
    buckets := make([][]int, len(nums)+1)
    for value, freq := range count {
        buckets[freq] = append(buckets[freq], value)
    }
    
    // Collect top k elements
    result := make([]int, 0, k)
    for i := len(buckets) - 1; i >= 0 && len(result) < k; i-- {
        result = append(result, buckets[i]...)
    }
    
    return result[:k]
}
```

#### **Using Quick Select**
```go
func topKFrequentQuickSelect(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    // Convert to slice of elements
    elements := make([]Element, 0, len(count))
    for value, freq := range count {
        elements = append(elements, Element{value, freq})
    }
    
    // Quick select to find kth largest
    quickSelect(elements, 0, len(elements)-1, k)
    
    // Return top k elements
    result := make([]int, k)
    for i := 0; i < k; i++ {
        result[i] = elements[i].value
    }
    
    return result
}

func quickSelect(elements []Element, left, right, k int) {
    if left >= right {
        return
    }
    
    pivotIndex := partition(elements, left, right)
    
    if pivotIndex == k-1 {
        return
    } else if pivotIndex > k-1 {
        quickSelect(elements, left, pivotIndex-1, k)
    } else {
        quickSelect(elements, pivotIndex+1, right, k)
    }
}

func partition(elements []Element, left, right int) int {
    pivot := elements[right].count
    i := left
    
    for j := left; j < right; j++ {
        if elements[j].count >= pivot {
            elements[i], elements[j] = elements[j], elements[i]
            i++
        }
    }
    
    elements[i], elements[right] = elements[right], elements[i]
    return i
}
```

#### **Using Sort**
```go
import "sort"

func topKFrequentSort(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    // Convert to slice of elements
    elements := make([]Element, 0, len(count))
    for value, freq := range count {
        elements = append(elements, Element{value, freq})
    }
    
    // Sort by frequency (descending)
    sort.Slice(elements, func(i, j int) bool {
        return elements[i].count > elements[j].count
    })
    
    // Return top k elements
    result := make([]int, k)
    for i := 0; i < k; i++ {
        result[i] = elements[i].value
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n log k) for heap, O(n) for bucket sort, O(n) average for quick select
- **Space Complexity:** O(n)