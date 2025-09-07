# Swap Nodes in Pairs

### Problem
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

**Example:**
```
Input: head = [1,2,3,4]
Output: [2,1,4,3]

Input: head = []
Output: []

Input: head = [1]
Output: [1]
```

### Golang Solution

```go
func swapPairs(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    
    for prev.Next != nil && prev.Next.Next != nil {
        first := prev.Next
        second := prev.Next.Next
        
        // Swap nodes
        prev.Next = second
        first.Next = second.Next
        second.Next = first
        
        // Move prev to the next pair
        prev = first
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func swapPairsRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    first := head
    second := head.Next
    
    first.Next = swapPairsRecursive(second.Next)
    second.Next = first
    
    return second
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive
