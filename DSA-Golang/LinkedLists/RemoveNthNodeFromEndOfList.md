# Remove Nth Node From End of List

### Problem
Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Example:**
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Input: head = [1], n = 1
Output: []

Input: head = [1,2], n = 1
Output: [1]
```

### Golang Solution

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Next: head}
    first := dummy
    second := dummy
    
    // Move first pointer n+1 steps ahead
    for i := 0; i <= n; i++ {
        first = first.Next
    }
    
    // Move both pointers until first reaches end
    for first != nil {
        first = first.Next
        second = second.Next
    }
    
    // Remove the nth node
    second.Next = second.Next.Next
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Two Pass Approach**
```go
func removeNthFromEndTwoPass(head *ListNode, n int) *ListNode {
    // First pass: count total nodes
    length := 0
    current := head
    for current != nil {
        length++
        current = current.Next
    }
    
    // Handle edge case
    if n == length {
        return head.Next
    }
    
    // Second pass: find and remove node
    current = head
    for i := 0; i < length-n-1; i++ {
        current = current.Next
    }
    
    current.Next = current.Next.Next
    return head
}
```

#### **Using Stack**
```go
func removeNthFromEndStack(head *ListNode, n int) *ListNode {
    stack := []*ListNode{}
    current := head
    
    // Push all nodes to stack
    for current != nil {
        stack = append(stack, current)
        current = current.Next
    }
    
    // Handle edge case
    if n == len(stack) {
        return head.Next
    }
    
    // Remove nth node from end
    targetIndex := len(stack) - n
    if targetIndex > 0 {
        stack[targetIndex-1].Next = stack[targetIndex].Next
    }
    
    return head
}
```

#### **Recursive Approach**
```go
func removeNthFromEndRecursive(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Next: head}
    removeNthFromEndHelper(dummy, n)
    return dummy.Next
}

func removeNthFromEndHelper(node *ListNode, n int) int {
    if node == nil {
        return 0
    }
    
    index := removeNthFromEndHelper(node.Next, n) + 1
    
    if index == n+1 {
        node.Next = node.Next.Next
    }
    
    return index
}
```

#### **Return Modified List with Count**
```go
func removeNthFromEndWithCount(head *ListNode, n int) (*ListNode, int) {
    dummy := &ListNode{Next: head}
    first := dummy
    second := dummy
    
    // Move first pointer n+1 steps ahead
    for i := 0; i <= n; i++ {
        first = first.Next
    }
    
    // Move both pointers until first reaches end
    for first != nil {
        first = first.Next
        second = second.Next
    }
    
    // Remove the nth node
    second.Next = second.Next.Next
    
    // Count remaining nodes
    count := 0
    current := dummy.Next
    for current != nil {
        count++
        current = current.Next
    }
    
    return dummy.Next, count
}
```

#### **Remove Multiple Nodes**
```go
func removeNthFromEndMultiple(head *ListNode, positions []int) *ListNode {
    for _, n := range positions {
        head = removeNthFromEnd(head, n)
    }
    return head
}
```

### Complexity
- **Time Complexity:** O(n) for optimal, O(n) for two pass
- **Space Complexity:** O(1) for optimal, O(n) for stack/recursive
