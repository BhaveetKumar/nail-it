# Linked Lists Pattern

> **Master linked list manipulation and algorithms with Go implementations**

## 📋 Problems

### **Basic Operations**
- [Reverse Linked List](./ReverseLinkedList.md) - Iterative and recursive approaches
- [Merge Two Sorted Lists](./MergeTwoSortedLists.md) - Merge sorted linked lists
- [Remove Duplicates from Sorted List](./RemoveDuplicatesFromSortedList.md) - Remove duplicates
- [Remove Linked List Elements](./RemoveLinkedListElements.md) - Remove specific values
- [Delete Node in a Linked List](./DeleteNodeInLinkedList.md) - Delete without head reference

### **Advanced Operations**
- [Add Two Numbers](./AddTwoNumbers.md) - Add numbers represented as linked lists
- [Swap Nodes in Pairs](./SwapNodesInPairs.md) - Swap adjacent nodes
- [Rotate List](./RotateList.md) - Rotate list to the right
- [Remove Nth Node From End of List](./RemoveNthNodeFromEnd.md) - Two pointers technique
- [Reorder List](./ReorderList.md) - Reorder list in specific pattern

### **Cycle Detection**
- [Linked List Cycle](./LinkedListCycle.md) - Floyd's cycle detection
- [Linked List Cycle II](./LinkedListCycleII.md) - Find cycle start node
- [Find the Duplicate Number](./FindTheDuplicateNumber.md) - Cycle detection application

### **Advanced Algorithms**
- [Copy List with Random Pointer](./CopyListWithRandomPointer.md) - Deep copy with random pointers
- [Sort List](./SortList.md) - Merge sort on linked list
- [Insertion Sort List](./InsertionSortList.md) - Insertion sort on linked list
- [Partition List](./PartitionList.md) - Partition around a value
- [Palindrome Linked List](./PalindromeLinkedList.md) - Check if list is palindrome

---

## 🎯 Key Concepts

### **Linked List Representation in Go**
```go
type ListNode struct {
    Val  int
    Next *ListNode
}

// Helper function to create a new node
func NewListNode(val int) *ListNode {
    return &ListNode{
        Val:  val,
        Next: nil,
    }
}
```

### **Common Patterns**
- **Two Pointers**: Fast and slow pointers for cycle detection
- **Dummy Head**: Simplify edge cases in list manipulation
- **Recursion**: Natural fit for linked list problems
- **Iteration**: More space-efficient than recursion

### **Performance Considerations**
- **Space Complexity**: Recursion uses O(n) stack space
- **Time Complexity**: Most operations are O(n)
- **Memory Management**: Go's garbage collector handles cleanup

---

## 🛠️ Go-Specific Tips

### **Linked List Traversal**
```go
// Iterative traversal
func traverseList(head *ListNode) {
    current := head
    for current != nil {
        // Process current node
        fmt.Println(current.Val)
        current = current.Next
    }
}

// Recursive traversal
func traverseListRecursive(head *ListNode) {
    if head == nil {
        return
    }
    // Process current node
    fmt.Println(head.Val)
    traverseListRecursive(head.Next)
}
```

### **Dummy Head Pattern**
```go
func removeElements(head *ListNode, val int) *ListNode {
    // Create dummy head to simplify edge cases
    dummy := &ListNode{Next: head}
    current := dummy
    
    for current.Next != nil {
        if current.Next.Val == val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

### **Two Pointers Technique**
```go
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

### **List Construction**
```go
func buildList(values []int) *ListNode {
    if len(values) == 0 {
        return nil
    }
    
    head := &ListNode{Val: values[0]}
    current := head
    
    for i := 1; i < len(values); i++ {
        current.Next = &ListNode{Val: values[i]}
        current = current.Next
    }
    
    return head
}
```

---

## 🎯 Interview Tips

### **How to Identify Linked List Problems**
1. **Traversal Problems**: Use iteration or recursion
2. **Reversal Problems**: Use iterative approach with three pointers
3. **Cycle Problems**: Use Floyd's cycle detection algorithm
4. **Merge Problems**: Use two pointers technique
5. **Partition Problems**: Use dummy heads for clean separation

### **Common Linked List Problem Patterns**
- **Reversal**: Reverse entire list or parts of list
- **Merging**: Combine two sorted lists
- **Cycle Detection**: Find cycles using two pointers
- **Partitioning**: Split list based on conditions
- **Sorting**: Apply sorting algorithms to linked lists

### **Optimization Tips**
- **Use dummy head**: Simplify edge case handling
- **Avoid recursion**: For large lists to prevent stack overflow
- **Two pointers**: For cycle detection and finding middle
- **In-place operations**: Modify existing list when possible
