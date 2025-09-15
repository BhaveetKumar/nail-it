# Merge k Sorted Lists

### Problem
You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

**Example:**
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Input: lists = []
Output: []

Input: lists = [[]]
Output: []
```

### Golang Solution

```go
import "container/heap"

type ListNode struct {
    Val  int
    Next *ListNode
}

type ListNodeHeap []*ListNode

func (h ListNodeHeap) Len() int           { return len(h) }
func (h ListNodeHeap) Less(i, j int) bool { return h[i].Val < h[j].Val }
func (h ListNodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *ListNodeHeap) Push(x interface{}) {
    *h = append(*h, x.(*ListNode))
}

func (h *ListNodeHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func mergeKLists(lists []*ListNode) *ListNode {
    h := &ListNodeHeap{}
    heap.Init(h)
    
    // Add all non-nil list heads to heap
    for _, list := range lists {
        if list != nil {
            heap.Push(h, list)
        }
    }
    
    dummy := &ListNode{}
    current := dummy
    
    for h.Len() > 0 {
        node := heap.Pop(h).(*ListNode)
        current.Next = node
        current = current.Next
        
        if node.Next != nil {
            heap.Push(h, node.Next)
        }
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Divide and Conquer**
```go
func mergeKListsDivideConquer(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    
    for len(lists) > 1 {
        var merged []*ListNode
        
        for i := 0; i < len(lists); i += 2 {
            if i+1 < len(lists) {
                merged = append(merged, mergeTwoLists(lists[i], lists[i+1]))
            } else {
                merged = append(merged, lists[i])
            }
        }
        
        lists = merged
    }
    
    return lists[0]
}

func mergeTwoLists(l1, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    current := dummy
    
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            current.Next = l1
            l1 = l1.Next
        } else {
            current.Next = l2
            l2 = l2.Next
        }
        current = current.Next
    }
    
    if l1 != nil {
        current.Next = l1
    } else {
        current.Next = l2
    }
    
    return dummy.Next
}
```

#### **Sequential Merge**
```go
func mergeKListsSequential(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    
    result := lists[0]
    
    for i := 1; i < len(lists); i++ {
        result = mergeTwoLists(result, lists[i])
    }
    
    return result
}
```

#### **Using Array and Sort**
```go
import "sort"

func mergeKListsArray(lists []*ListNode) *ListNode {
    var values []int
    
    for _, list := range lists {
        current := list
        for current != nil {
            values = append(values, current.Val)
            current = current.Next
        }
    }
    
    sort.Ints(values)
    
    dummy := &ListNode{}
    current := dummy
    
    for _, val := range values {
        current.Next = &ListNode{Val: val}
        current = current.Next
    }
    
    return dummy.Next
}
```

#### **Return with Statistics**
```go
type MergeStats struct {
    TotalNodes    int
    TotalLists    int
    MinValue      int
    MaxValue      int
    AvgValue      float64
    MergedList    *ListNode
}

func mergeKListsWithStats(lists []*ListNode) MergeStats {
    if len(lists) == 0 {
        return MergeStats{TotalLists: 0}
    }
    
    var values []int
    totalNodes := 0
    minValue := math.MaxInt32
    maxValue := math.MinInt32
    sum := 0
    
    for _, list := range lists {
        current := list
        for current != nil {
            values = append(values, current.Val)
            totalNodes++
            if current.Val < minValue {
                minValue = current.Val
            }
            if current.Val > maxValue {
                maxValue = current.Val
            }
            sum += current.Val
            current = current.Next
        }
    }
    
    sort.Ints(values)
    
    dummy := &ListNode{}
    current := dummy
    
    for _, val := range values {
        current.Next = &ListNode{Val: val}
        current = current.Next
    }
    
    return MergeStats{
        TotalNodes: totalNodes,
        TotalLists: len(lists),
        MinValue:   minValue,
        MaxValue:   maxValue,
        AvgValue:   float64(sum) / float64(totalNodes),
        MergedList: dummy.Next,
    }
}
```

#### **Return All Possible Merges**
```go
func allPossibleMerges(lists []*ListNode) []*ListNode {
    if len(lists) == 0 {
        return []*ListNode{}
    }
    
    var result []*ListNode
    
    // Generate all permutations of merge orders
    var permute func([]*ListNode, int)
    permute = func(arr []*ListNode, start int) {
        if start == len(arr) {
            merged := mergeKListsSequential(arr)
            result = append(result, merged)
            return
        }
        
        for i := start; i < len(arr); i++ {
            arr[start], arr[i] = arr[i], arr[start]
            permute(arr, start+1)
            arr[start], arr[i] = arr[i], arr[start]
        }
    }
    
    permute(lists, 0)
    return result
}
```

### Complexity
- **Time Complexity:** O(n log k) for heap approach, O(nk) for sequential merge
- **Space Complexity:** O(k) for heap, O(1) for divide and conquer