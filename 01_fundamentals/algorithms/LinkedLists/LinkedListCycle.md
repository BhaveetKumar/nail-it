# Linked List Cycle

### Problem
Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. Note that `pos` is not passed as a parameter.

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.

**Example:**
```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
```

**Constraints:**
- The number of the nodes in the list is in the range [0, 10⁴]
- -10⁵ ≤ Node.val ≤ 10⁵
- pos is -1 or a valid index in the linked-list

### Explanation

#### **Floyd's Cycle Detection Algorithm (Tortoise and Hare)**
- Use two pointers: slow (tortoise) and fast (hare)
- Slow pointer moves one step at a time
- Fast pointer moves two steps at a time
- If there's a cycle, the fast pointer will eventually meet the slow pointer
- Time Complexity: O(n)
- Space Complexity: O(1)

#### **Hash Set Approach**
- Use a hash set to store visited nodes
- If we encounter a node that's already in the set, there's a cycle
- Time Complexity: O(n)
- Space Complexity: O(n)

### Dry Run

**Input:** `head = [3,2,0,-4]` with cycle at position 1

#### **Floyd's Algorithm**

| Step | Slow Position | Fast Position | Slow Value | Fast Value | Action |
|------|---------------|---------------|------------|------------|---------|
| 0 | 3 | 3 | 3 | 3 | Start |
| 1 | 2 | 0 | 2 | 0 | Move pointers |
| 2 | 0 | 2 | 0 | 2 | Move pointers |
| 3 | -4 | -4 | -4 | -4 | **Cycle detected!** |

**Result:** `true`

### Complexity
- **Time Complexity:** O(n) - In worst case, fast pointer traverses the list twice
- **Space Complexity:** O(1) - Only using two pointers

### Golang Solution

#### **Floyd's Cycle Detection Algorithm**
```go
type ListNode struct {
    Val  int
    Next *ListNode
}

func hasCycle(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return false
    }
    
    slow := head
    fast := head.Next
    
    for fast != nil && fast.Next != nil {
        if slow == fast {
            return true
        }
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    return false
}
```

#### **Hash Set Approach**
```go
func hasCycleHashSet(head *ListNode) bool {
    if head == nil {
        return false
    }
    
    visited := make(map[*ListNode]bool)
    current := head
    
    for current != nil {
        if visited[current] {
            return true
        }
        visited[current] = true
        current = current.Next
    }
    
    return false
}
```

### Alternative Solutions

#### **Optimized Floyd's Algorithm**
```go
func hasCycleOptimized(head *ListNode) bool {
    if head == nil {
        return false
    }
    
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            return true
        }
    }
    
    return false
}
```

#### **Find Cycle Start Node**
```go
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    // First phase: detect if cycle exists
    slow := head
    fast := head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            // Cycle detected, find the start
            slow = head
            for slow != fast {
                slow = slow.Next
                fast = fast.Next
            }
            return slow
        }
    }
    
    return nil
}
```

#### **Count Cycle Length**
```go
func cycleLength(head *ListNode) int {
    if head == nil {
        return 0
    }
    
    slow := head
    fast := head
    
    // Detect cycle
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        if slow == fast {
            // Count cycle length
            length := 1
            fast = fast.Next
            
            for slow != fast {
                fast = fast.Next
                length++
            }
            
            return length
        }
    }
    
    return 0
}
```

### Notes / Variations

#### **Related Problems**
- **Linked List Cycle II**: Find the start node of the cycle
- **Find the Duplicate Number**: Use cycle detection on array
- **Happy Number**: Detect cycle in number sequence
- **Circular Array Loop**: Detect cycle in circular array
- **Floyd's Algorithm**: Mathematical proof of correctness

#### **ICPC Insights**
- **Floyd's Algorithm**: Optimal for cycle detection
- **Space Efficiency**: O(1) space complexity
- **Mathematical Proof**: Understand why the algorithm works
- **Edge Cases**: Handle empty list and single node

#### **Go-Specific Optimizations**
```go
// Use pointer comparison for efficiency
func hasCycle(head *ListNode) bool {
    if head == nil {
        return false
    }
    
    slow, fast := head, head
    
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
        
        // Pointer comparison is O(1)
        if slow == fast {
            return true
        }
    }
    
    return false
}

// Use struct embedding for additional functionality
type ListNode struct {
    Val  int
    Next *ListNode
}

func (l *ListNode) HasCycle() bool {
    return hasCycle(l)
}
```

#### **Real-World Applications**
- **Memory Management**: Detect memory leaks in circular references
- **Network Protocols**: Detect infinite loops in routing
- **Game Development**: Detect infinite loops in game states
- **Compiler Design**: Detect infinite recursion

### Testing

```go
func TestHasCycle(t *testing.T) {
    tests := []struct {
        name     string
        values   []int
        pos      int
        expected bool
    }{
        {
            name:     "Cycle at position 1",
            values:   []int{3, 2, 0, -4},
            pos:      1,
            expected: true,
        },
        {
            name:     "Cycle at position 0",
            values:   []int{1, 2},
            pos:      0,
            expected: true,
        },
        {
            name:     "No cycle",
            values:   []int{1},
            pos:      -1,
            expected: false,
        },
        {
            name:     "Empty list",
            values:   []int{},
            pos:      -1,
            expected: false,
        },
    }
    
    for _, test := range tests {
        t.Run(test.name, func(t *testing.T) {
            head := buildListWithCycle(test.values, test.pos)
            result := hasCycle(head)
            
            if result != test.expected {
                t.Errorf("hasCycle() = %v, expected %v", result, test.expected)
            }
        })
    }
}

// Helper functions
func buildListWithCycle(values []int, pos int) *ListNode {
    if len(values) == 0 {
        return nil
    }
    
    nodes := make([]*ListNode, len(values))
    
    // Create nodes
    for i, val := range values {
        nodes[i] = &ListNode{Val: val}
    }
    
    // Connect nodes
    for i := 0; i < len(values)-1; i++ {
        nodes[i].Next = nodes[i+1]
    }
    
    // Create cycle if pos is valid
    if pos >= 0 && pos < len(values) {
        nodes[len(values)-1].Next = nodes[pos]
    }
    
    return nodes[0]
}
```

### Visualization

```
List with cycle: 3 -> 2 -> 0 -> -4
                        ^         |
                        |         |
                        -----------

Floyd's Algorithm:
Step 1: slow=3, fast=3
Step 2: slow=2, fast=0
Step 3: slow=0, fast=2
Step 4: slow=-4, fast=-4 ← Cycle detected!

Mathematical Proof:
- If there's a cycle, fast pointer will eventually meet slow pointer
- Distance between pointers decreases by 1 each step
- When they meet, we've found the cycle
```

### Performance Comparison

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| Floyd's Algorithm | O(n) | O(1) | Space efficient | Complex logic |
| Hash Set | O(n) | O(n) | Simple logic | Extra space needed |
| Brute Force | O(n²) | O(1) | Easy to understand | Inefficient |

**Recommendation**: Use Floyd's algorithm for optimal performance, hash set for simplicity.
