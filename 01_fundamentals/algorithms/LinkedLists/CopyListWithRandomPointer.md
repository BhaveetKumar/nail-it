# Copy List with Random Pointer

### Problem
A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a deep copy of the list. The deep copy should consist of exactly `n` brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

**Example:**
```
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
```

### Golang Solution

```go
type Node struct {
    Val    int
    Next   *Node
    Random *Node
}

func copyRandomList(head *Node) *Node {
    if head == nil {
        return nil
    }
    
    // Step 1: Create new nodes and insert them after original nodes
    current := head
    for current != nil {
        newNode := &Node{Val: current.Val}
        newNode.Next = current.Next
        current.Next = newNode
        current = newNode.Next
    }
    
    // Step 2: Set random pointers for new nodes
    current = head
    for current != nil {
        if current.Random != nil {
            current.Next.Random = current.Random.Next
        }
        current = current.Next.Next
    }
    
    // Step 3: Separate original and new lists
    current = head
    newHead := head.Next
    newCurrent := newHead
    
    for current != nil {
        current.Next = current.Next.Next
        if newCurrent.Next != nil {
            newCurrent.Next = newCurrent.Next.Next
        }
        current = current.Next
        newCurrent = newCurrent.Next
    }
    
    return newHead
}
```

### Alternative Solutions

#### **Hash Map Approach**
```go
func copyRandomListHashMap(head *Node) *Node {
    if head == nil {
        return nil
    }
    
    nodeMap := make(map[*Node]*Node)
    
    // Create new nodes
    current := head
    for current != nil {
        nodeMap[current] = &Node{Val: current.Val}
        current = current.Next
    }
    
    // Set next and random pointers
    current = head
    for current != nil {
        nodeMap[current].Next = nodeMap[current.Next]
        nodeMap[current].Random = nodeMap[current.Random]
        current = current.Next
    }
    
    return nodeMap[head]
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for in-place, O(n) for hash map
