# Merge Two Sorted Lists

### Problem
You are given the heads of two sorted linked lists `list1` and `list2`.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

**Example:**
```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Input: list1 = [], list2 = []
Output: []

Input: list1 = [], list2 = [0]
Output: [0]
```

### Golang Solution

```go
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummy := &ListNode{}
    current := dummy
    
    for list1 != nil && list2 != nil {
        if list1.Val <= list2.Val {
            current.Next = list1
            list1 = list1.Next
        } else {
            current.Next = list2
            list2 = list2.Next
        }
        current = current.Next
    }
    
    // Attach remaining nodes
    if list1 != nil {
        current.Next = list1
    } else {
        current.Next = list2
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func mergeTwoListsRecursive(list1 *ListNode, list2 *ListNode) *ListNode {
    if list1 == nil {
        return list2
    }
    if list2 == nil {
        return list1
    }
    
    if list1.Val <= list2.Val {
        list1.Next = mergeTwoListsRecursive(list1.Next, list2)
        return list1
    } else {
        list2.Next = mergeTwoListsRecursive(list1, list2.Next)
        return list2
    }
}
```

#### **In-Place Merging**
```go
func mergeTwoListsInPlace(list1 *ListNode, list2 *ListNode) *ListNode {
    if list1 == nil {
        return list2
    }
    if list2 == nil {
        return list1
    }
    
    var head *ListNode
    if list1.Val <= list2.Val {
        head = list1
        list1 = list1.Next
    } else {
        head = list2
        list2 = list2.Next
    }
    
    current := head
    
    for list1 != nil && list2 != nil {
        if list1.Val <= list2.Val {
            current.Next = list1
            list1 = list1.Next
        } else {
            current.Next = list2
            list2 = list2.Next
        }
        current = current.Next
    }
    
    if list1 != nil {
        current.Next = list1
    } else {
        current.Next = list2
    }
    
    return head
}
```

#### **Using Priority Queue**
```go
import "container/heap"

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

func mergeTwoListsPriorityQueue(list1 *ListNode, list2 *ListNode) *ListNode {
    h := &ListNodeHeap{}
    heap.Init(h)
    
    // Add all nodes to heap
    for list1 != nil {
        heap.Push(h, list1)
        list1 = list1.Next
    }
    
    for list2 != nil {
        heap.Push(h, list2)
        list2 = list2.Next
    }
    
    dummy := &ListNode{}
    current := dummy
    
    for h.Len() > 0 {
        node := heap.Pop(h).(*ListNode)
        current.Next = node
        current = current.Next
    }
    
    current.Next = nil
    return dummy.Next
}
```

#### **Merge with Count**
```go
func mergeTwoListsWithCount(list1 *ListNode, list2 *ListNode) (*ListNode, int) {
    dummy := &ListNode{}
    current := dummy
    count := 0
    
    for list1 != nil && list2 != nil {
        if list1.Val <= list2.Val {
            current.Next = list1
            list1 = list1.Next
        } else {
            current.Next = list2
            list2 = list2.Next
        }
        current = current.Next
        count++
    }
    
    // Attach remaining nodes
    for list1 != nil {
        current.Next = list1
        current = current.Next
        list1 = list1.Next
        count++
    }
    
    for list2 != nil {
        current.Next = list2
        current = current.Next
        list2 = list2.Next
        count++
    }
    
    return dummy.Next, count
}
```

#### **Merge Multiple Lists**
```go
func mergeKLists(lists []*ListNode) *ListNode {
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
```

### Complexity
- **Time Complexity:** O(n + m) where n and m are lengths of the two lists
- **Space Complexity:** O(1) for iterative, O(n + m) for recursive