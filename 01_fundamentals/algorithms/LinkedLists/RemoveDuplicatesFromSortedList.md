---
# Auto-generated front matter
Title: Removeduplicatesfromsortedlist
LastUpdated: 2025-11-06T20:45:58.750271
Tags: []
Status: draft
---

# Remove Duplicates from Sorted List

### Problem
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

**Example:**
```
Input: head = [1,1,2]
Output: [1,2]

Input: head = [1,1,2,3,3]
Output: [1,2,3]
```

### Golang Solution

```go
func deleteDuplicates(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    current := head
    
    for current.Next != nil {
        if current.Val == current.Next.Val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    return head
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
        return deleteDuplicatesRecursive(head.Next)
    }
    
    head.Next = deleteDuplicatesRecursive(head.Next)
    return head
}
```

#### **Using Dummy Node**
```go
func deleteDuplicatesDummy(head *ListNode) *ListNode {
    dummy := &ListNode{Next: head}
    current := dummy
    
    for current.Next != nil && current.Next.Next != nil {
        if current.Next.Val == current.Next.Next.Val {
            val := current.Next.Val
            for current.Next != nil && current.Next.Val == val {
                current.Next = current.Next.Next
            }
        } else {
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

#### **Remove All Duplicates (Not Just Consecutive)**
```go
func deleteAllDuplicates(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    // Count frequency of each value
    freq := make(map[int]int)
    current := head
    
    for current != nil {
        freq[current.Val]++
        current = current.Next
    }
    
    // Create new list with unique values
    dummy := &ListNode{}
    tail := dummy
    current = head
    
    for current != nil {
        if freq[current.Val] == 1 {
            tail.Next = &ListNode{Val: current.Val}
            tail = tail.Next
        }
        current = current.Next
    }
    
    return dummy.Next
}
```

#### **Return Count of Removed Duplicates**
```go
func deleteDuplicatesWithCount(head *ListNode) (*ListNode, int) {
    if head == nil {
        return nil, 0
    }
    
    removedCount := 0
    current := head
    
    for current.Next != nil {
        if current.Val == current.Next.Val {
            current.Next = current.Next.Next
            removedCount++
        } else {
            current = current.Next
        }
    }
    
    return head, removedCount
}
```

#### **Remove Duplicates from Unsorted List**
```go
func deleteDuplicatesUnsorted(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }
    
    seen := make(map[int]bool)
    dummy := &ListNode{Next: head}
    current := dummy
    
    for current.Next != nil {
        if seen[current.Next.Val] {
            current.Next = current.Next.Next
        } else {
            seen[current.Next.Val] = true
            current = current.Next
        }
    }
    
    return dummy.Next
}
```

#### **Return List with Duplicate Info**
```go
type DuplicateInfo struct {
    List     *ListNode
    Original int
    Removed  int
}

func deleteDuplicatesWithInfo(head *ListNode) DuplicateInfo {
    if head == nil {
        return DuplicateInfo{List: nil, Original: 0, Removed: 0}
    }
    
    originalCount := 0
    current := head
    
    // Count original nodes
    for current != nil {
        originalCount++
        current = current.Next
    }
    
    // Remove duplicates
    current = head
    for current.Next != nil {
        if current.Val == current.Next.Val {
            current.Next = current.Next.Next
        } else {
            current = current.Next
        }
    }
    
    // Count remaining nodes
    remainingCount := 0
    current = head
    for current != nil {
        remainingCount++
        current = current.Next
    }
    
    return DuplicateInfo{
        List:     head,
        Original: originalCount,
        Removed:  originalCount - remainingCount,
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(1) for in-place, O(n) for hash map approach