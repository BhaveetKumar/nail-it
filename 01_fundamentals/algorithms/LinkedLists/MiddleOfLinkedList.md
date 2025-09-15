# Middle of the Linked List

### Problem
Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

**Example:**
```
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.
```

### Golang Solution

```go
func middleNode(head *ListNode) *ListNode {
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    return slow
}
```

### Alternative Solutions

#### **Using Two Passes**
```go
func middleNodeTwoPass(head *ListNode) *ListNode {
    // First pass: count nodes
    count := 0
    current := head
    
    for current != nil {
        count++
        current = current.Next
    }
    
    // Second pass: find middle
    middle := count / 2
    current = head
    
    for i := 0; i < middle; i++ {
        current = current.Next
    }
    
    return current
}
```

#### **Using Array**
```go
func middleNodeArray(head *ListNode) *ListNode {
    var nodes []*ListNode
    current := head
    
    for current != nil {
        nodes = append(nodes, current)
        current = current.Next
    }
    
    return nodes[len(nodes)/2]
}
```

#### **Return First Middle Node**
```go
func middleNodeFirst(head *ListNode) *ListNode {
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    return slow
}
```

#### **Return Middle Value**
```go
func middleValue(head *ListNode) int {
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    return slow.Val
}
```

#### **Return All Middle Nodes**
```go
func allMiddleNodes(head *ListNode) []*ListNode {
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    var result []*ListNode
    
    // If odd number of nodes, return single middle
    if fast == nil {
        result = append(result, slow)
    } else {
        // If even number of nodes, return both middle nodes
        result = append(result, slow)
        result = append(result, slow.Next)
    }
    
    return result
}
```

#### **Return Middle with Position Info**
```go
type MiddleResult struct {
    Node     *ListNode
    Position int
    Total    int
}

func middleNodeWithInfo(head *ListNode) MiddleResult {
    // Count total nodes
    total := 0
    current := head
    
    for current != nil {
        total++
        current = current.Next
    }
    
    // Find middle node
    middle := total / 2
    current = head
    
    for i := 0; i < middle; i++ {
        current = current.Next
    }
    
    return MiddleResult{
        Node:     current,
        Position: middle,
        Total:    total,
    }
}
```

### Complexity
- **Time Complexity:** O(n) for all approaches
- **Space Complexity:** O(1) for two pointers, O(n) for array approach
