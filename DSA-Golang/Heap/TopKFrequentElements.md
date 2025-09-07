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
    num   int
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
    
    for num, freq := range count {
        heap.Push(h, Element{num: num, count: freq})
    }
    
    result := make([]int, k)
    for i := 0; i < k; i++ {
        element := heap.Pop(h).(Element)
        result[i] = element.num
    }
    
    return result
}
```

### Alternative Solutions

#### **Bucket Sort**
```go
func topKFrequentBucketSort(nums []int, k int) []int {
    count := make(map[int]int)
    for _, num := range nums {
        count[num]++
    }
    
    buckets := make([][]int, len(nums)+1)
    for num, freq := range count {
        buckets[freq] = append(buckets[freq], num)
    }
    
    result := make([]int, 0, k)
    for i := len(buckets) - 1; i >= 0 && len(result) < k; i-- {
        result = append(result, buckets[i]...)
    }
    
    return result[:k]
}
```

### Complexity
- **Time Complexity:** O(n log n) for heap, O(n) for bucket sort
- **Space Complexity:** O(n)
