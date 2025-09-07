# Partition List

### Problem
Given the head of a linked list and a value `x`, partition it such that all nodes less than `x` come before nodes greater than or equal to `x`.

You should preserve the original relative order of the nodes in each of the two partitions.

**Example:**
```
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Input: head = [2,1], x = 2
Output: [1,2]
```

### Golang Solution

```go
func partition(head *ListNode, x int) *ListNode {
    // Create dummy nodes for two partitions
    beforeHead := &ListNode{}
    afterHead := &ListNode{}
    
    before := beforeHead
    after := afterHead
    
    current := head
    
    for current != nil {
        if current.Val < x {
            before.Next = current
            before = before.Next
        } else {
            after.Next = current
            after = after.Next
        }
        current = current.Next
    }
    
    // Connect the two partitions
    after.Next = nil
    before.Next = afterHead.Next
    
    return beforeHead.Next
}
```

### Alternative Solutions

#### **In-Place Approach**
```go
func partitionInPlace(head *ListNode, x int) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    current := head
    
    // Find the first node >= x
    for current != nil && current.Val < x {
        prev = current
        current = current.Next
    }
    
    // If all nodes are < x, return original list
    if current == nil {
        return head
    }
    
    // Move nodes < x to the beginning
    for current.Next != nil {
        if current.Next.Val < x {
            // Remove the node
            temp := current.Next
            current.Next = current.Next.Next
            
            // Insert at the beginning
            temp.Next = prev.Next
            prev.Next = temp
            prev = temp
        } else {
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

#### **Using Arrays**
```go
func partitionArrays(head *ListNode, x int) *ListNode {
    if head == nil {
        return nil
    }
    
    var less, greater []int
    current := head
    
    // Collect values
    for current != nil {
        if current.Val < x {
            less = append(less, current.Val)
        } else {
            greater = append(greater, current.Val)
        }
        current = current.Next
    }
    
    // Create new list
    dummy := &ListNode{}
    current = dummy
    
    // Add less values
    for _, val := range less {
        current.Next = &ListNode{Val: val}
        current = current.Next
    }
    
    // Add greater values
    for _, val := range greater {
        current.Next = &ListNode{Val: val}
        current = current.Next
    }
    
    return dummy.Next
}
```

#### **Recursive Approach**
```go
func partitionRecursive(head *ListNode, x int) *ListNode {
    if head == nil {
        return nil
    }
    
    if head.Val < x {
        head.Next = partitionRecursive(head.Next, x)
        return head
    } else {
        // Find the first node < x in the rest
        rest := partitionRecursive(head.Next, x)
        
        if rest == nil || rest.Val >= x {
            head.Next = rest
            return head
        }
        
        // Move the smaller node to the front
        temp := rest
        for temp.Next != nil && temp.Next.Val < x {
            temp = temp.Next
        }
        
        head.Next = temp.Next
        temp.Next = head
        
        return rest
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for optimal, O(n) for arrays/recursive
