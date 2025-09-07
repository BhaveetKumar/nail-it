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

#### **Using Hash Set**
```go
func getIntersectionNodeHash(headA, headB *ListNode) *ListNode {
    visited := make(map[*ListNode]bool)
    
    // Traverse first list
    current := headA
    for current != nil {
        visited[current] = true
        current = current.Next
    }
    
    // Traverse second list
    current = headB
    for current != nil {
        if visited[current] {
            return current
        }
        current = current.Next
    }
    
    return nil
}
```

#### **Using Length Difference**
```go
func getIntersectionNodeLength(headA, headB *ListNode) *ListNode {
    lenA := getLength(headA)
    lenB := getLength(headB)
    
    // Move longer list forward
    for lenA > lenB {
        headA = headA.Next
        lenA--
    }
    
    for lenB > lenA {
        headB = headB.Next
        lenB--
    }
    
    // Find intersection
    for headA != headB {
        headA = headA.Next
        headB = headB.Next
    }
    
    return headA
}

func getLength(head *ListNode) int {
    length := 0
    for head != nil {
        length++
        head = head.Next
    }
    return length
}
```

#### **Return with Distance Info**
```go
type IntersectionResult struct {
    Node     *ListNode
    DistanceA int
    DistanceB int
    Found    bool
}

func getIntersectionWithInfo(headA, headB *ListNode) IntersectionResult {
    if headA == nil || headB == nil {
        return IntersectionResult{Found: false}
    }
    
    a, b := headA, headB
    distanceA, distanceB := 0, 0
    
    for a != b {
        if a == nil {
            a = headB
            distanceA = 0
        } else {
            a = a.Next
            distanceA++
        }
        
        if b == nil {
            b = headA
            distanceB = 0
        } else {
            b = b.Next
            distanceB++
        }
    }
    
    return IntersectionResult{
        Node:      a,
        DistanceA: distanceA,
        DistanceB: distanceB,
        Found:     a != nil,
    }
}
```

#### **Return All Common Nodes**
```go
func getAllCommonNodes(headA, headB *ListNode) []*ListNode {
    var common []*ListNode
    visited := make(map[*ListNode]bool)
    
    // Traverse first list
    current := headA
    for current != nil {
        visited[current] = true
        current = current.Next
    }
    
    // Traverse second list
    current = headB
    for current != nil {
        if visited[current] {
            common = append(common, current)
        }
        current = current.Next
    }
    
    return common
}
```

#### **Check if Lists Intersect**
```go
func hasIntersection(headA, headB *ListNode) bool {
    if headA == nil || headB == nil {
        return false
    }
    
    // Find tail of first list
    tailA := headA
    for tailA.Next != nil {
        tailA = tailA.Next
    }
    
    // Find tail of second list
    tailB := headB
    for tailB.Next != nil {
        tailB = tailB.Next
    }
    
    return tailA == tailB
}
```

#### **Return Intersection Statistics**
```go
type IntersectionStats struct {
    HasIntersection bool
    IntersectionNode *ListNode
    LengthA         int
    LengthB         int
    CommonLength    int
    DistanceA       int
    DistanceB       int
}

func intersectionStatistics(headA, headB *ListNode) IntersectionStats {
    if headA == nil || headB == nil {
        return IntersectionStats{HasIntersection: false}
    }
    
    lenA := getLength(headA)
    lenB := getLength(headB)
    
    // Check if lists intersect
    tailA := headA
    for tailA.Next != nil {
        tailA = tailA.Next
    }
    
    tailB := headB
    for tailB.Next != nil {
        tailB = tailB.Next
    }
    
    if tailA != tailB {
        return IntersectionStats{
            HasIntersection: false,
            LengthA:         lenA,
            LengthB:         lenB,
        }
    }
    
    // Find intersection
    intersection := getIntersectionNode(headA, headB)
    
    // Calculate distances
    distanceA := 0
    current := headA
    for current != intersection {
        distanceA++
        current = current.Next
    }
    
    distanceB := 0
    current = headB
    for current != intersection {
        distanceB++
        current = current.Next
    }
    
    commonLength := lenA + lenB - distanceA - distanceB
    
    return IntersectionStats{
        HasIntersection:   true,
        IntersectionNode:  intersection,
        LengthA:           lenA,
        LengthB:           lenB,
        CommonLength:      commonLength,
        DistanceA:         distanceA,
        DistanceB:         distanceB,
    }
}
```

### Complexity
- **Time Complexity:** O(m + n) where m and n are the lengths of the lists
- **Space Complexity:** O(1) for two pointers, O(m + n) for hash set