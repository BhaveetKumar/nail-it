# Remove Duplicates from Sorted List

### Problem
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

**Example:**
```
Input: head = [1,1,2]
Output: [1,2]

Input: head = [1,1,2,3,3]
Output: [1,2,3]
```

### Golang Solution

```go
func deleteDuplicates(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    current := head
    
    for current.Next != nil {
        if current.Val == current.Next.Val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    return head
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func deleteDuplicatesRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    head.Next = deleteDuplicatesRecursive(head.Next)
    
    if head.Val == head.Next.Val {
        return head.Next
    }
    
    return head
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive
