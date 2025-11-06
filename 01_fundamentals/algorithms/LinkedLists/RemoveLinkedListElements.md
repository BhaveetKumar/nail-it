---
# Auto-generated front matter
Title: Removelinkedlistelements
LastUpdated: 2025-11-06T20:45:58.753170
Tags: []
Status: draft
---

# Remove Linked List Elements

### Problem
Given the head of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return the new head.

**Example:**
```
Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]

Input: head = [], val = 1
Output: []

Input: head = [7,7,7,7], val = 7
Output: []
```

### Golang Solution

```go
func removeElements(head *ListNode, val int) *ListNode {
    // Handle case where head needs to be removed
    for head != nil && head.Val == val {
        head = head.Next
    }
    
    if head == nil {
        return nil
    }
    
    current := head
    
    for current.Next != nil {
        if current.Next.Val == val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    return head
}
```

### Alternative Solutions

#### **Using Dummy Node**
```go
func removeElementsDummy(head *ListNode, val int) *ListNode {
    dummy := &ListNode{Next: head}
    current := dummy
    
    for current.Next != nil {
        if current.Next.Val == val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

#### **Recursive Approach**
```go
func removeElementsRecursive(head *ListNode, val int) *ListNode {
    if head == nil {
        return nil
    }
    
    head.Next = removeElementsRecursive(head.Next, val)
    
    if head.Val == val {
        return head.Next
    }
    
    return head
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive
