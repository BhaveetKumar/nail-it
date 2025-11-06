---
# Auto-generated front matter
Title: Reorderlist
LastUpdated: 2025-11-06T20:45:58.753287
Tags: []
Status: draft
---

# Reorder List

### Problem
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

**Example:**
```
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

### Golang Solution

```go
func reorderList(head *ListNode) {
    if head == nil || head.Next == nil {
        return
    }
    
    // Step 1: Find the middle of the list
    slow, fast := head, head
    for fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    // Step 2: Reverse the second half
    second := reverseList(slow.Next)
    slow.Next = nil
    
    // Step 3: Merge the two halves
    first := head
    for second != nil {
        temp1 := first.Next
        temp2 := second.Next
        
        first.Next = second
        second.Next = temp1
        
        first = temp1
        second = temp2
    }
}

func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    current := head
    
    for current != nil {
        next := current.Next
        current.Next = prev
        prev = current
        current = next
    }
    
    return prev
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
