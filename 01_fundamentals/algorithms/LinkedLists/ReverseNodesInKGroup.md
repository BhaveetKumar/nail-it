---
# Auto-generated front matter
Title: Reversenodesinkgroup
LastUpdated: 2025-11-06T20:45:58.752411
Tags: []
Status: draft
---

# Reverse Nodes in k-Group

### Problem
Given the head of a linked list, reverse the nodes of the list `k` at a time, and return the modified list.

`k` is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of `k` then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

**Example:**
```
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
```

### Golang Solution

```go
func reverseKGroup(head *ListNode, k int) *ListNode {
    if head == nil || k == 1 {
        return head
    }
    
    // Check if we have at least k nodes
    count := 0
    current := head
    for current != nil && count < k {
        current = current.Next
        count++
    }
    
    if count == k {
        // Reverse the first k nodes
        current = reverseKGroup(current, k)
        
        // Reverse the current group
        for count > 0 {
            next := head.Next
            head.Next = current
            current = head
            head = next
            count--
        }
        head = current
    }
    
    return head
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
func reverseKGroupIterative(head *ListNode, k int) *ListNode {
    if head == nil || k == 1 {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    
    for {
        // Check if we have at least k nodes
        count := 0
        current := prev.Next
        for current != nil && count < k {
            current = current.Next
            count++
        }
        
        if count < k {
            break
        }
        
        // Reverse k nodes
        tail := prev.Next
        for count > 1 {
            next := tail.Next
            tail.Next = next.Next
            next.Next = prev.Next
            prev.Next = next
            count--
        }
        
        prev = tail
    }
    
    return dummy.Next
}
```

#### **Using Stack**
```go
func reverseKGroupStack(head *ListNode, k int) *ListNode {
    if head == nil || k == 1 {
        return head
    }
    
    dummy := &ListNode{}
    tail := dummy
    stack := make([]*ListNode, 0, k)
    current := head
    
    for current != nil {
        // Collect k nodes
        for len(stack) < k && current != nil {
            stack = append(stack, current)
            current = current.Next
        }
        
        if len(stack) == k {
            // Reverse and append
            for i := len(stack) - 1; i >= 0; i-- {
                tail.Next = stack[i]
                tail = tail.Next
            }
            stack = stack[:0]
        } else {
            // Append remaining nodes as is
            for _, node := range stack {
                tail.Next = node
                tail = tail.Next
            }
        }
    }
    
    tail.Next = nil
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(k) for stack, O(n/k) for recursive
