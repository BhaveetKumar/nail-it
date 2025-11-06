---
# Auto-generated front matter
Title: Reverselinkedlistii
LastUpdated: 2025-11-06T20:45:58.752839
Tags: []
Status: draft
---

# Reverse Linked List II

### Problem
Given the head of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return the reversed list.

**Example:**
```
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Input: head = [5], left = 1, right = 1
Output: [5]
```

### Golang Solution

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
    if head == nil || left == right {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    
    // Move to the node before the reversal
    for i := 0; i < left-1; i++ {
        prev = prev.Next
    }
    
    // Start reversing
    current := prev.Next
    for i := 0; i < right-left; i++ {
        next := current.Next
        current.Next = next.Next
        next.Next = prev.Next
        prev.Next = next
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func reverseBetweenRecursive(head *ListNode, left int, right int) *ListNode {
    if left == 1 {
        return reverseN(head, right)
    }
    
    head.Next = reverseBetweenRecursive(head.Next, left-1, right-1)
    return head
}

func reverseN(head *ListNode, n int) *ListNode {
    if n == 1 {
        return head
    }
    
    newHead := reverseN(head.Next, n-1)
    head.Next.Next = head
    head.Next = nil
    
    return newHead
}
```

#### **Two Pass Approach**
```go
func reverseBetweenTwoPass(head *ListNode, left int, right int) *ListNode {
    if head == nil || left == right {
        return head
    }
    
    dummy := &ListNode{Next: head}
    
    // Find the node before left
    prev := dummy
    for i := 0; i < left-1; i++ {
        prev = prev.Next
    }
    
    // Find the node at right
    rightNode := prev
    for i := 0; i < right-left+1; i++ {
        rightNode = rightNode.Next
    }
    
    // Extract the sublist
    leftNode := prev.Next
    curr := rightNode.Next
    
    // Disconnect the sublist
    prev.Next = nil
    rightNode.Next = nil
    
    // Reverse the sublist
    reverseList(leftNode)
    
    // Reconnect
    prev.Next = rightNode
    leftNode.Next = curr
    
    return dummy.Next
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

#### **Using Stack**
```go
func reverseBetweenStack(head *ListNode, left int, right int) *ListNode {
    if head == nil || left == right {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    
    // Move to the node before left
    for i := 0; i < left-1; i++ {
        prev = prev.Next
    }
    
    // Collect nodes to reverse
    stack := []*ListNode{}
    current := prev.Next
    
    for i := 0; i < right-left+1; i++ {
        stack = append(stack, current)
        current = current.Next
    }
    
    // Reconnect in reverse order
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        prev.Next = node
        prev = prev.Next
    }
    
    prev.Next = current
    return dummy.Next
}
```

#### **Iterative with Three Pointers**
```go
func reverseBetweenThreePointers(head *ListNode, left int, right int) *ListNode {
    if head == nil || left == right {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    
    // Move to the node before left
    for i := 0; i < left-1; i++ {
        prev = prev.Next
    }
    
    // Reverse the sublist
    current := prev.Next
    var next *ListNode
    
    for i := 0; i < right-left; i++ {
        next = current.Next
        current.Next = next.Next
        next.Next = prev.Next
        prev.Next = next
    }
    
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive/stack