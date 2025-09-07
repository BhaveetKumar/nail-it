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

type MedianFinder struct {
    maxHeap *IntHeap // for smaller half
    minHeap *IntHeap // for larger half
}

func Constructor() MedianFinder {
    maxHeap := &IntHeap{}
    minHeap := &IntHeap{}
    heap.Init(maxHeap)
    heap.Init(minHeap)
    
    return MedianFinder{
        maxHeap: maxHeap,
        minHeap: minHeap,
    }
}

func (this *MedianFinder) AddNum(num int) {
    if this.maxHeap.Len() == 0 || num <= (*this.maxHeap)[0] {
        heap.Push(this.maxHeap, -num) // Use negative for max heap
    } else {
        heap.Push(this.minHeap, num)
    }
    
    // Balance heaps
    if this.maxHeap.Len() > this.minHeap.Len()+1 {
        val := heap.Pop(this.maxHeap).(int)
        heap.Push(this.minHeap, -val)
    } else if this.minHeap.Len() > this.maxHeap.Len()+1 {
        val := heap.Pop(this.minHeap).(int)
        heap.Push(this.maxHeap, -val)
    }
}

func (this *MedianFinder) FindMedian() float64 {
    if this.maxHeap.Len() > this.minHeap.Len() {
        return float64(-(*this.maxHeap)[0])
    } else if this.minHeap.Len() > this.maxHeap.Len() {
        return float64((*this.minHeap)[0])
    } else {
        return float64(-(*this.maxHeap)[0]+(*this.minHeap)[0]) / 2.0
    }
}
```

### Alternative Solutions

#### **Using Two Heaps with Custom Max Heap**
```go
type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] }
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

type MedianFinderCustom struct {
    maxHeap *MaxHeap
    minHeap *IntHeap
}

func NewMedianFinderCustom() *MedianFinderCustom {
    maxHeap := &MaxHeap{}
    minHeap := &IntHeap{}
    heap.Init(maxHeap)
    heap.Init(minHeap)
    
    return &MedianFinderCustom{
        maxHeap: maxHeap,
        minHeap: minHeap,
    }
}

func (this *MedianFinderCustom) AddNum(num int) {
    if this.maxHeap.Len() == 0 || num <= (*this.maxHeap)[0] {
        heap.Push(this.maxHeap, num)
    } else {
        heap.Push(this.minHeap, num)
    }
    
    // Balance heaps
    if this.maxHeap.Len() > this.minHeap.Len()+1 {
        val := heap.Pop(this.maxHeap).(int)
        heap.Push(this.minHeap, val)
    } else if this.minHeap.Len() > this.maxHeap.Len()+1 {
        val := heap.Pop(this.minHeap).(int)
        heap.Push(this.maxHeap, val)
    }
}

func (this *MedianFinderCustom) FindMedian() float64 {
    if this.maxHeap.Len() > this.minHeap.Len() {
        return float64((*this.maxHeap)[0])
    } else if this.minHeap.Len() > this.maxHeap.Len() {
        return float64((*this.minHeap)[0])
    } else {
        return float64((*this.maxHeap)[0]+(*this.minHeap)[0]) / 2.0
    }
}
```

#### **Using Sorted Array**
```go
type MedianFinderArray struct {
    nums []int
}

func NewMedianFinderArray() *MedianFinderArray {
    return &MedianFinderArray{nums: []int{}}
}

func (this *MedianFinderArray) AddNum(num int) {
    // Insert in sorted order
    i := 0
    for i < len(this.nums) && this.nums[i] < num {
        i++
    }
    
    this.nums = append(this.nums, 0)
    copy(this.nums[i+1:], this.nums[i:])
    this.nums[i] = num
}

func (this *MedianFinderArray) FindMedian() float64 {
    n := len(this.nums)
    if n%2 == 1 {
        return float64(this.nums[n/2])
    } else {
        return float64(this.nums[n/2-1]+this.nums[n/2]) / 2.0
    }
}
```

#### **Using Binary Search Tree**
```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
    Count int
}

type MedianFinderBST struct {
    root *TreeNode
    size int
}

func NewMedianFinderBST() *MedianFinderBST {
    return &MedianFinderBST{}
}

func (this *MedianFinderBST) AddNum(num int) {
    this.root = this.insert(this.root, num)
    this.size++
}

func (this *MedianFinderBST) insert(root *TreeNode, val int) *TreeNode {
    if root == nil {
        return &TreeNode{Val: val, Count: 1}
    }
    
    if val <= root.Val {
        root.Left = this.insert(root.Left, val)
    } else {
        root.Right = this.insert(root.Right, val)
    }
    
    root.Count++
    return root
}

func (this *MedianFinderBST) FindMedian() float64 {
    if this.size%2 == 1 {
        return float64(this.findKth(this.root, this.size/2+1))
    } else {
        return float64(this.findKth(this.root, this.size/2)+this.findKth(this.root, this.size/2+1)) / 2.0
    }
}

func (this *MedianFinderBST) findKth(root *TreeNode, k int) int {
    leftCount := 0
    if root.Left != nil {
        leftCount = root.Left.Count
    }
    
    if k <= leftCount {
        return this.findKth(root.Left, k)
    } else if k == leftCount+1 {
        return root.Val
    } else {
        return this.findKth(root.Right, k-leftCount-1)
    }
}
```

#### **Return All Statistics**
```go
type MedianFinderStats struct {
    maxHeap *IntHeap
    minHeap *IntHeap
    count   int
    sum     int64
}

func NewMedianFinderStats() *MedianFinderStats {
    maxHeap := &IntHeap{}
    minHeap := &IntHeap{}
    heap.Init(maxHeap)
    heap.Init(minHeap)
    
    return &MedianFinderStats{
        maxHeap: maxHeap,
        minHeap: minHeap,
    }
}

func (this *MedianFinderStats) AddNum(num int) {
    this.count++
    this.sum += int64(num)
    
    if this.maxHeap.Len() == 0 || num <= -(*this.maxHeap)[0] {
        heap.Push(this.maxHeap, -num)
    } else {
        heap.Push(this.minHeap, num)
    }
    
    // Balance heaps
    if this.maxHeap.Len() > this.minHeap.Len()+1 {
        val := heap.Pop(this.maxHeap).(int)
        heap.Push(this.minHeap, -val)
    } else if this.minHeap.Len() > this.maxHeap.Len()+1 {
        val := heap.Pop(this.minHeap).(int)
        heap.Push(this.maxHeap, -val)
    }
}

func (this *MedianFinderStats) FindMedian() float64 {
    if this.maxHeap.Len() > this.minHeap.Len() {
        return float64(-(*this.maxHeap)[0])
    } else if this.minHeap.Len() > this.maxHeap.Len() {
        return float64((*this.minHeap)[0])
    } else {
        return float64(-(*this.maxHeap)[0]+(*this.minHeap)[0]) / 2.0
    }
}

func (this *MedianFinderStats) GetCount() int {
    return this.count
}

func (this *MedianFinderStats) GetSum() int64 {
    return this.sum
}

func (this *MedianFinderStats) GetAverage() float64 {
    if this.count == 0 {
        return 0
    }
    return float64(this.sum) / float64(this.count)
}
```

### Complexity
- **Time Complexity:** O(log n) for addNum, O(1) for findMedian
- **Space Complexity:** O(n)