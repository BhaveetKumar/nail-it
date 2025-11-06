---
# Auto-generated front matter
Title: Addtwonumbers
LastUpdated: 2025-11-06T20:45:58.750963
Tags: []
Status: draft
---

# Add Two Numbers

### Problem
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example:**
```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Input: l1 = [0], l2 = [0]
Output: [0]

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
```

### Golang Solution

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    current := dummy
    carry := 0
    
    for l1 != nil || l2 != nil || carry != 0 {
        sum := carry
        
        if l1 != nil {
            sum += l1.Val
            l1 = l1.Next
        }
        
        if l2 != nil {
            sum += l2.Val
            l2 = l2.Next
        }
        
        carry = sum / 10
        current.Next = &ListNode{Val: sum % 10}
        current = current.Next
    }
    
    return dummy.Next
}
```

### Alternative Solutions

#### **Recursive Approach**
```go
func addTwoNumbersRecursive(l1 *ListNode, l2 *ListNode) *ListNode {
    return addTwoNumbersHelper(l1, l2, 0)
}

func addTwoNumbersHelper(l1 *ListNode, l2 *ListNode, carry int) *ListNode {
    if l1 == nil && l2 == nil && carry == 0 {
        return nil
    }
    
    sum := carry
    var next1, next2 *ListNode
    
    if l1 != nil {
        sum += l1.Val
        next1 = l1.Next
    }
    
    if l2 != nil {
        sum += l2.Val
        next2 = l2.Next
    }
    
    return &ListNode{
        Val:  sum % 10,
        Next: addTwoNumbersHelper(next1, next2, sum/10),
    }
}
```

#### **In-Place Modification**
```go
func addTwoNumbersInPlace(l1 *ListNode, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    
    head := l1
    carry := 0
    
    for l1 != nil || l2 != nil {
        sum := carry
        
        if l1 != nil {
            sum += l1.Val
        }
        
        if l2 != nil {
            sum += l2.Val
            l2 = l2.Next
        }
        
        carry = sum / 10
        l1.Val = sum % 10
        
        if l1.Next == nil && (l2 != nil || carry > 0) {
            l1.Next = &ListNode{}
        }
        
        l1 = l1.Next
    }
    
    return head
}
```

#### **Using Stack**
```go
func addTwoNumbersStack(l1 *ListNode, l2 *ListNode) *ListNode {
    stack1 := []int{}
    stack2 := []int{}
    
    // Push all values to stacks
    for l1 != nil {
        stack1 = append(stack1, l1.Val)
        l1 = l1.Next
    }
    
    for l2 != nil {
        stack2 = append(stack2, l2.Val)
        l2 = l2.Next
    }
    
    // Add numbers
    var result *ListNode
    carry := 0
    
    for len(stack1) > 0 || len(stack2) > 0 || carry > 0 {
        sum := carry
        
        if len(stack1) > 0 {
            sum += stack1[len(stack1)-1]
            stack1 = stack1[:len(stack1)-1]
        }
        
        if len(stack2) > 0 {
            sum += stack2[len(stack2)-1]
            stack2 = stack2[:len(stack2)-1]
        }
        
        carry = sum / 10
        result = &ListNode{Val: sum % 10, Next: result}
    }
    
    return result
}
```

#### **Return as Integer (Not Recommended for Large Numbers)**
```go
func addTwoNumbersAsInt(l1 *ListNode, l2 *ListNode) *ListNode {
    num1 := listToInt(l1)
    num2 := listToInt(l2)
    sum := num1 + num2
    
    return intToList(sum)
}

func listToInt(head *ListNode) int {
    num := 0
    multiplier := 1
    
    for head != nil {
        num += head.Val * multiplier
        multiplier *= 10
        head = head.Next
    }
    
    return num
}

func intToList(num int) *ListNode {
    if num == 0 {
        return &ListNode{Val: 0}
    }
    
    dummy := &ListNode{}
    current := dummy
    
    for num > 0 {
        current.Next = &ListNode{Val: num % 10}
        current = current.Next
        num /= 10
    }
    
    return dummy.Next
}
```

### Complexity
- **Time Complexity:** O(max(m, n)) where m and n are lengths of the two lists
- **Space Complexity:** O(max(m, n)) for new list, O(1) for in-place