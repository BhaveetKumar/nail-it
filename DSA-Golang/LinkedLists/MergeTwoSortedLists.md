# Merge Two Sorted Lists

### Problem
You are given the heads of two sorted linked lists `list1` and `list2`.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

**Example:**
```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
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

### Complexity
- **Time Complexity:** O(n + m)
- **Space Complexity:** O(1)
