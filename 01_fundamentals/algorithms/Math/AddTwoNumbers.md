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
    return addTwoNumbersWithCarry(l1, l2, 0)
}

func addTwoNumbersWithCarry(l1 *ListNode, l2 *ListNode, carry int) *ListNode {
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
        Next: addTwoNumbersWithCarry(next1, next2, sum/10),
    }
}
```

#### **Convert to Numbers (Not Recommended for Large Numbers)**
```go
func addTwoNumbersConvert(l1 *ListNode, l2 *ListNode) *ListNode {
    num1 := listToNumber(l1)
    num2 := listToNumber(l2)
    sum := num1 + num2
    
    return numberToList(sum)
}

func listToNumber(head *ListNode) int {
    num := 0
    multiplier := 1
    
    for head != nil {
        num += head.Val * multiplier
        multiplier *= 10
        head = head.Next
    }
    
    return num
}

func numberToList(num int) *ListNode {
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
- **Space Complexity:** O(max(m, n))
