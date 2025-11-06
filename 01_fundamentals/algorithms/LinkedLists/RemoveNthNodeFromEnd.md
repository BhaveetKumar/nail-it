---
# Auto-generated front matter
Title: Removenthnodefromend
LastUpdated: 2025-11-06T20:45:58.750054
Tags: []
Status: draft
---

# Remove Nth Node From End of List

### Problem
Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Example:**
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Input: head = [1], n = 1
Output: []
```

### Golang Solution

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    dummy := &ListNode{Next: head}
    first := dummy
    second := dummy
    
    // Move first pointer n+1 steps ahead
    for i := 0; i <= n; i++ {
        first = first.Next
    }
    
    // Move both pointers until first reaches end
    for first != nil {
        first = first.Next
        second = second.Next
    }
    
    // Remove the nth node
    second.Next = second.Next.Next
    
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
