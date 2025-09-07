# Remove Duplicates from Sorted List II

### Problem
Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.

**Example:**
```
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]

Input: head = [1,1,1,2,3]
Output: [2,3]
```

### Golang Solution

```go
func deleteDuplicates(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    current := head
    
    for current != nil {
        // Skip all duplicates
        if current.Next != nil && current.Val == current.Next.Val {
            val := current.Val
            for current != nil && current.Val == val {
                current = current.Next
            }
            prev.Next = current
        } else {
            prev = current
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func deleteDuplicatesRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    if head.Val == head.Next.Val {
        val := head.Val
        for head != nil && head.Val == val {
            head = head.Next
        }
        return deleteDuplicatesRecursive(head)
    }
    
    head.Next = deleteDuplicatesRecursive(head.Next)
    return head
}
```

#### **Two Pointer Approach**
```go
func deleteDuplicatesTwoPointer(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    slow := dummy
    fast := head
    
    for fast != nil {
        if fast.Next != nil && fast.Val == fast.Next.Val {
            val := fast.Val
            for fast != nil && fast.Val == val {
                fast = fast.Next
            }
            slow.Next = fast
        } else {
            slow = slow.Next
            fast = fast.Next
        }
    }
    
    return dummy.Next
}
```

#### **Using Hash Map**
```go
func deleteDuplicatesHashMap(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    // Count occurrences
    count := make(map[int]int)
    current := head
    
    for current != nil {
        count[current.Val]++
        current = current.Next
    }
    
    // Create new list with non-duplicates
    dummy := &ListNode{}
    tail := dummy
    current = head
    
    for current != nil {
        if count[current.Val] == 1 {
            tail.Next = &ListNode{Val: current.Val}
            tail = tail.Next
        }
        current = current.Next
    }
    
    return dummy.Next
}
```

#### **In-Place with Counter**
```go
func deleteDuplicatesInPlace(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    dummy := &ListNode{Next: head}
    prev := dummy
    current := head
    
    for current != nil {
        // Count duplicates
        count := 1
        temp := current.Next
        
        for temp != nil && temp.Val == current.Val {
            count++
            temp = temp.Next
        }
        
        if count > 1 {
            // Skip all duplicates
            prev.Next = temp
            current = temp
        } else {
            prev = current
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for iterative, O(n) for recursive/hash map