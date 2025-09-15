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
func findClosestElements(arr []int, k int, x int) []int {
    left, right := 0, len(arr)-k
    
    for left < right {
        mid := left + (right-left)/2
        
        if x-arr[mid] > arr[mid+k]-x {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return arr[left : left+k]
}
```

### Alternative Solutions

#### **Using Two Pointers**
```go
func findClosestElementsTwoPointers(arr []int, k int, x int) []int {
    left, right := 0, len(arr)-1
    
    for right-left+1 > k {
        if abs(arr[left]-x) <= abs(arr[right]-x) {
            right--
        } else {
            left++
        }
    }
    
    return arr[left : right+1]
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```

#### **Using Heap**
```go
import "container/heap"

type Element struct {
    value int
    diff  int
}

type MaxHeap []Element

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].diff > h[j].diff }
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

func findClosestElementsHeap(arr []int, k int, x int) []int {
    h := &MaxHeap{}
    heap.Init(h)
    
    for _, num := range arr {
        diff := abs(num - x)
        element := Element{value: num, diff: diff}
        
        if h.Len() < k {
            heap.Push(h, element)
        } else if diff < (*h)[0].diff {
            heap.Pop(h)
            heap.Push(h, element)
        }
    }
    
    var result []int
    for h.Len() > 0 {
        element := heap.Pop(h).(Element)
        result = append(result, element.value)
    }
    
    sort.Ints(result)
    return result
}
```

#### **Using Sorting**
```go
import "sort"

func findClosestElementsSort(arr []int, k int, x int) []int {
    // Create a copy and sort by distance to x
    sorted := make([]int, len(arr))
    copy(sorted, arr)
    
    sort.Slice(sorted, func(i, j int) bool {
        diffI := abs(sorted[i] - x)
        diffJ := abs(sorted[j] - x)
        
        if diffI == diffJ {
            return sorted[i] < sorted[j]
        }
        return diffI < diffJ
    })
    
    // Take first k elements and sort them
    result := sorted[:k]
    sort.Ints(result)
    
    return result
}
```

#### **Return with Distances**
```go
type ClosestElement struct {
    Value int
    Diff  int
}

func findClosestElementsWithDistances(arr []int, k int, x int) []ClosestElement {
    left, right := 0, len(arr)-k
    
    for left < right {
        mid := left + (right-left)/2
        
        if x-arr[mid] > arr[mid+k]-x {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    var result []ClosestElement
    for i := left; i < left+k; i++ {
        result = append(result, ClosestElement{
            Value: arr[i],
            Diff:  abs(arr[i] - x),
        })
    }
    
    return result
}
```

#### **Find K Closest with Custom Comparator**
```go
func findClosestElementsCustom(arr []int, k int, x int) []int {
    // Find the position where x should be inserted
    pos := binarySearch(arr, x)
    
    left, right := pos, pos
    
    // Expand window to include k elements
    for right-left < k {
        if left == 0 {
            right++
        } else if right == len(arr) {
            left--
        } else if abs(arr[left-1]-x) <= abs(arr[right]-x) {
            left--
        } else {
            right++
        }
    }
    
    return arr[left:right]
}

func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)
    
    for left < right {
        mid := left + (right-left)/2
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid
        }
    }
    
    return left
}
```

### Complexity
- **Time Complexity:** O(log n + k) for binary search, O(n log n) for sorting
- **Space Complexity:** O(1) for binary search, O(n) for heap/sorting
