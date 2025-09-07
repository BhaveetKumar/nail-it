# Palindrome Linked List

### Problem
Given the head of a singly linked list, return `true` if it is a palindrome or `false` otherwise.

**Example:**
```
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
```

### Golang Solution

```go
func isPalindrome(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return true
    }
    
    // Find the middle of the linked list
    slow, fast := head, head
    for fast.Next != nil && fast.Next.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    // Reverse the second half
    secondHalf := reverseList(slow.Next)
    firstHalf := head
    
    // Compare both halves
    for secondHalf != nil {
        if firstHalf.Val != secondHalf.Val {
            return false
        }
        firstHalf = firstHalf.Next
        secondHalf = secondHalf.Next
    }
    
    return true
}

func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    current := head
    
    for current != nil {
        next := current.Next
        current.Next = prev
        prev = current
        current = next
    }
    
    return prev
}
```

### Alternative Solutions

#### **Using Array**
```go
func isPalindromeArray(head *ListNode) bool {
    values := []int{}
    current := head
    
    for current != nil {
        values = append(values, current.Val)
        current = current.Next
    }
    
    left, right := 0, len(values)-1
    for left < right {
        if values[left] != values[right] {
            return false
        }
        left++
        right--
    }
    
    return true
}
```

#### **Recursive Approach**
```go
func isPalindromeRecursive(head *ListNode) bool {
    front := head
    
    var recursivelyCheck func(*ListNode) bool
    recursivelyCheck = func(current *ListNode) bool {
        if current != nil {
            if !recursivelyCheck(current.Next) {
                return false
            }
            if current.Val != front.Val {
                return false
            }
            front = front.Next
        }
        return true
    }
    
    return recursivelyCheck(head)
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for two pointers, O(n) for array/recursive
