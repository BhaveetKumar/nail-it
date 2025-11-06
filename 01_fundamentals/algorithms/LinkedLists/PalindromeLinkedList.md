---
# Auto-generated front matter
Title: Palindromelinkedlist
LastUpdated: 2025-11-06T20:45:58.753407
Tags: []
Status: draft
---

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
    
    // Find middle
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    // Reverse second half
    secondHalf := reverseList(slow)
    
    // Compare first half with reversed second half
    firstHalf := head
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
    var values []int
    
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

#### **Using Stack**
```go
func isPalindromeStack(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return true
    }
    
    // Find middle
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    
    // Push first half to stack
    stack := []int{}
    current := head
    for current != slow {
        stack = append(stack, current.Val)
        current = current.Next
    }
    
    // Skip middle node if odd length
    if fast != nil {
        slow = slow.Next
    }
    
    // Compare with second half
    for slow != nil {
        if len(stack) == 0 || stack[len(stack)-1] != slow.Val {
            return false
        }
        stack = stack[:len(stack)-1]
        slow = slow.Next
    }
    
    return true
}
```

#### **Return with Values**
```go
type PalindromeResult struct {
    IsPalindrome bool
    Values      []int
    Reversed    []int
}

func isPalindromeWithValues(head *ListNode) PalindromeResult {
    var values []int
    
    current := head
    for current != nil {
        values = append(values, current.Val)
        current = current.Next
    }
    
    reversed := make([]int, len(values))
    for i, val := range values {
        reversed[len(values)-1-i] = val
    }
    
    isPalindrome := true
    for i := 0; i < len(values); i++ {
        if values[i] != reversed[i] {
            isPalindrome = false
            break
        }
    }
    
    return PalindromeResult{
        IsPalindrome: isPalindrome,
        Values:      values,
        Reversed:    reversed,
    }
}
```

#### **Return All Palindromes**
```go
func findAllPalindromes(head *ListNode) [][]int {
    var palindromes [][]int
    var current []int
    
    var dfs func(*ListNode)
    dfs = func(node *ListNode) {
        if node == nil {
            if len(current) > 0 && isPalindromeArray(createList(current)) {
                palindrome := make([]int, len(current))
                copy(palindrome, current)
                palindromes = append(palindromes, palindrome)
            }
            return
        }
        
        current = append(current, node.Val)
        dfs(node.Next)
        current = current[:len(current)-1]
    }
    
    dfs(head)
    return palindromes
}

func createList(values []int) *ListNode {
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

#### **Return Longest Palindrome**
```go
func longestPalindrome(head *ListNode) []int {
    var longest []int
    var current []int
    
    var dfs func(*ListNode)
    dfs = func(node *ListNode) {
        if node == nil {
            if len(current) > len(longest) && isPalindromeArray(createList(current)) {
                longest = make([]int, len(current))
                copy(longest, current)
            }
            return
        }
        
        current = append(current, node.Val)
        dfs(node.Next)
        current = current[:len(current)-1]
    }
    
    dfs(head)
    return longest
}
```

#### **Return Palindrome Statistics**
```go
type PalindromeStats struct {
    IsPalindrome    bool
    Length         int
    MiddleValue    int
    FirstHalf      []int
    SecondHalf     []int
    PalindromeCount int
}

func palindromeStatistics(head *ListNode) PalindromeStats {
    if head == nil {
        return PalindromeStats{IsPalindrome: true}
    }
    
    var values []int
    current := head
    for current != nil {
        values = append(values, current.Val)
        current = current.Next
    }
    
    length := len(values)
    isPalindrome := true
    var firstHalf, secondHalf []int
    
    for i := 0; i < length/2; i++ {
        if values[i] != values[length-1-i] {
            isPalindrome = false
        }
        firstHalf = append(firstHalf, values[i])
        secondHalf = append(secondHalf, values[length-1-i])
    }
    
    middleValue := -1
    if length%2 == 1 {
        middleValue = values[length/2]
    }
    
    return PalindromeStats{
        IsPalindrome:    isPalindrome,
        Length:         length,
        MiddleValue:    middleValue,
        FirstHalf:      firstHalf,
        SecondHalf:     secondHalf,
        PalindromeCount: 1,
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(1) for in-place, O(n) for array/stack