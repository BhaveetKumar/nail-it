# Intersection of Two Linked Lists

### Problem
Given the heads of two singly linked-lists `headA` and `headB`, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return `null`.

**Example:**
```
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
```

### Golang Solution

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    if headA == nil || headB == nil {
        return nil
    }
    
    a, b := headA, headB
    
    for a != b {
        if a == nil {
            a = headB
        } else {
            a = a.Next
        }
        
        if b == nil {
            b = headA
        } else {
            b = b.Next
        }
    }
    
    return a
}
```

### Alternative Solutions

#### **Hash Set Approach**
```go
func getIntersectionNodeHashSet(headA, headB *ListNode) *ListNode {
    visited := make(map[*ListNode]bool)
    
    // Traverse list A and mark all nodes as visited
    for headA != nil {
        visited[headA] = true
        headA = headA.Next
    }
    
    // Traverse list B and check if any node is already visited
    for headB != nil {
        if visited[headB] {
            return headB
        }
        headB = headB.Next
    }
    
    return nil
}
```

### Complexity
- **Time Complexity:** O(m + n)
- **Space Complexity:** O(1) for two pointers, O(m + n) for hash set
