---
# Auto-generated front matter
Title: Swapnodesinpairs
LastUpdated: 2025-11-06T20:45:58.753975
Tags: []
Status: draft
---

# Swap Nodes in Pairs

### Problem
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed).

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
        
        // Move to next pair
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
    
    // Swap the first two nodes
    first.Next = swapPairsRecursive(second.Next)
    second.Next = first
    
    return second
}
```

#### **Iterative with Three Pointers**
```go
func swapPairsIterative(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    current := head
    
    for current != nil && current.Next != nil {
        next := current.Next
        nextNext := next.Next
        
        // Swap current and next
        prev.Next = next
        next.Next = current
        current.Next = nextNext
        
        // Move pointers
        prev = current
        current = nextNext
    }
    
    return dummy.Next
}
```

#### **Using Stack**
```go
func swapPairsStack(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    stack := []*ListNode{}
    current := head
    
    // Push nodes in pairs
    for current != nil && current.Next != nil {
        stack = append(stack, current, current.Next)
        current = current.Next.Next
    }
    
    // Handle odd length list
    if current != nil {
        stack = append(stack, current)
    }
    
    // Rebuild list with swapped pairs
    dummy := &ListNode{}
    prev := dummy
    
    for i := 0; i < len(stack); i += 2 {
        if i+1 < len(stack) {
            // Swap pair
            prev.Next = stack[i+1]
            stack[i+1].Next = stack[i]
            prev = stack[i]
        } else {
            // Single node
            prev.Next = stack[i]
        }
    }
    
    // Set last node's next to nil
    if len(stack) > 0 {
        stack[0].Next = nil
    }
    
    return dummy.Next
}
```

#### **In-Place with Value Swapping (Not Recommended)**
```go
func swapPairsValueSwap(head *ListNode) *ListNode {
    current := head
    
    for current != nil && current.Next != nil {
        // Swap values
        current.Val, current.Next.Val = current.Next.Val, current.Val
        current = current.Next.Next
    }
    
    return head
}
```

#### **Return All Possible Swaps**
```go
func swapPairsAll(head *ListNode) []*ListNode {
    if head == nil || head.Next == nil {
        return []*ListNode{head}
    }
    
    var results []*ListNode
    
    // Original list
    results = append(results, copyList(head))
    
    // Swapped list
    swapped := swapPairs(head)
    results = append(results, swapped)
    
    return results
}

func copyList(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    dummy := &ListNode{}
    current := dummy
    original := head
    
    for original != nil {
        current.Next = &ListNode{Val: original.Val}
        current = current.Next
        original = original.Next
    }
    
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive