# Find K Closest Elements

### Problem
Given a sorted integer array `arr`, two integers `k` and `x`, return the `k` closest integers to `x` in the array. The result should also be sorted in ascending order.

An integer `a` is closer to `x` than an integer `b` if:
- `|a - x| < |b - x|`, or
- `|a - x| == |b - x|` and `a < b`

**Example:**
```
Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]

Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]
```

### Golang Solution

```go
import "sort"

func findClosestElements(arr []int, k int, x int) []int {
    // Sort by distance from x, then by value
    sort.Slice(arr, func(i, j int) bool {
        diffI := abs(arr[i] - x)
        diffJ := abs(arr[j] - x)
        if diffI == diffJ {
            return arr[i] < arr[j]
        }
        return diffI < diffJ
    })
    
    // Take first k elements and sort them
    result := arr[:k]
    sort.Ints(result)
    
    return result
}

func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```

### Alternative Solutions

#### **Binary Search + Two Pointers**
```go
func findClosestElementsBinarySearch(arr []int, k int, x int) []int {
    left := 0
    right := len(arr) - k
    
    // Binary search for the left boundary
    for left < right {
        mid := left + (right-left)/2
        
        // Compare distances
        if x-arr[mid] > arr[mid+k]-x {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return arr[left : left+k]
}
```

#### **Using Priority Queue**
```go
import "container/heap"

type Element struct {
    value    int
    distance int
}

type ElementHeap []Element

func (h ElementHeap) Len() int { return len(h) }
func (h ElementHeap) Less(i, j int) bool {
    if h[i].distance == h[j].distance {
        return h[i].value < h[j].value
    }
    return h[i].distance < h[j].distance
}
func (h ElementHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *ElementHeap) Push(x interface{}) {
    *h = append(*h, x.(Element))
}

func (h *ElementHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func findClosestElementsPQ(arr []int, k int, x int) []int {
    h := &ElementHeap{}
    heap.Init(h)
    
    for _, num := range arr {
        heap.Push(h, Element{
            value:    num,
            distance: abs(num - x),
        })
    }
    
    var result []int
    for i := 0; i < k; i++ {
        element := heap.Pop(h).(Element)
        result = append(result, element.value)
    }
    
    sort.Ints(result)
    return result
}
```

#### **Two Pointers from Center**
```go
func findClosestElementsTwoPointers(arr []int, k int, x int) []int {
    // Find the closest element to x
    left := 0
    right := len(arr) - 1
    
    for right-left > 1 {
        mid := left + (right-left)/2
        if arr[mid] < x {
            left = mid
        } else {
            right = mid
        }
    }
    
    // Expand from the closest element
    for right-left+1 < k {
        if left == 0 {
            right++
        } else if right == len(arr)-1 {
            left--
        } else if x-arr[left-1] <= arr[right+1]-x {
            left--
        } else {
            right++
        }
    }
    
    return arr[left : right+1]
}
```

#### **Sliding Window**
```go
func findClosestElementsSlidingWindow(arr []int, k int, x int) []int {
    left := 0
    right := len(arr) - 1
    
    // Shrink window to size k
    for right-left+1 > k {
        if abs(arr[left]-x) <= abs(arr[right]-x) {
            right--
        } else {
            left++
        }
    }
    
    return arr[left : right+1]
}
```

### Complexity
- **Time Complexity:** O(n log n) for sorting, O(log n + k) for binary search
- **Space Complexity:** O(1) for binary search, O(n) for priority queue
