# Reverse Linked List

### Problem

Given the head of a singly linked list, reverse the list, and return the reversed list.

**Example:**

```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Input: head = [1,2]
Output: [2,1]

Input: head = []
Output: []
```

**Constraints:**

- The number of nodes in the list is the range [0, 5000]
- -5000 ≤ Node.val ≤ 5000

### Explanation

#### **Iterative Approach**

- Use three pointers: `prev`, `current`, and `next`
- At each step, reverse the link between `current` and `prev`
- Move all pointers forward
- Time Complexity: O(n)
- Space Complexity: O(1)

#### **Recursive Approach**

- Recursively reverse the rest of the list
- Then reverse the link between current node and the next node
- Time Complexity: O(n)
- Space Complexity: O(n) for recursion stack

### Dry Run

**Input:** `head = [1,2,3,4,5]`

#### **Iterative Approach**

| Step | prev | current | next | Action              |
| ---- | ---- | ------- | ---- | ------------------- |
| 0    | nil  | 1       | 2    | Initialize          |
| 1    | 1    | 2       | 3    | current.Next = prev |
| 2    | 2    | 3       | 4    | current.Next = prev |
| 3    | 3    | 4       | 5    | current.Next = prev |
| 4    | 4    | 5       | nil  | current.Next = prev |
| 5    | 5    | nil     | -    | current.Next = prev |

**Result:** `[5,4,3,2,1]`

### Complexity

- **Time Complexity:** O(n) - Single pass through the list
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

#### **Iterative Solution**

```go
type ListNode struct {
    Val  int
    Next *ListNode
}

func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    current := head

    for current != nil {
        // Store the next node before reversing
        next := current.Next

        // Reverse the link
        current.Next = prev

        // Move pointers forward
        prev = current
        current = next
    }

    // prev is now the new head
    return prev
}
```

#### **Recursive Solution**

```go
func reverseListRecursive(head *ListNode) *ListNode {
    // Base case: empty list or single node
    if head == nil || head.Next == nil {
        return head
    }

    // Recursively reverse the rest of the list
    newHead := reverseListRecursive(head.Next)

    // Reverse the link between current node and next node
    head.Next.Next = head
    head.Next = nil

    return newHead
}
```

### Alternative Solutions

#### **Using Stack**

```go
func reverseListStack(head *ListNode) *ListNode {
    if head == nil {
        return nil
    }

    // Push all nodes onto stack
    stack := []*ListNode{}
    current := head

    for current != nil {
        stack = append(stack, current)
        current = current.Next
    }

    // Pop nodes and reverse links
    newHead := stack[len(stack)-1]
    current = newHead

    for i := len(stack) - 2; i >= 0; i-- {
        current.Next = stack[i]
        current = current.Next
    }

    current.Next = nil
    return newHead
}
```

#### **Reverse in Groups**

```go
func reverseListInGroups(head *ListNode, k int) *ListNode {
    if head == nil || k <= 1 {
        return head
    }

    // Count nodes in current group
    count := 0
    current := head

    for current != nil && count < k {
        current = current.Next
        count++
    }

    // If we have k nodes, reverse them
    if count == k {
        // Reverse first k nodes
        current = reverseListInGroups(current, k)

        // Reverse current group
        for count > 0 {
            next := head.Next
            head.Next = current
            current = head
            head = next
            count--
        }

        head = current
    }

    return head
}
```

### Notes / Variations

#### **Related Problems**

- **Reverse Linked List II**: Reverse a portion of the list
- **Reverse Nodes in k-Group**: Reverse every k consecutive nodes
- **Palindrome Linked List**: Check if list is palindrome
- **Reorder List**: Reorder list in specific pattern
- **Swap Nodes in Pairs**: Swap adjacent nodes

#### **ICPC Insights**

- **Memory Efficiency**: Iterative approach uses O(1) space
- **Stack Overflow**: Avoid recursion for very long lists
- **Edge Cases**: Handle empty list and single node
- **Pointer Management**: Be careful with pointer assignments

#### **Go-Specific Optimizations**

```go
// Use pointer to pointer for cleaner code
func reverseList(head **ListNode) {
    var prev *ListNode
    current := *head

    for current != nil {
        next := current.Next
        current.Next = prev
        prev = current
        current = next
    }

    *head = prev
}

// Use struct embedding for additional functionality
type ListNode struct {
    Val  int
    Next *ListNode
}

func (l *ListNode) Reverse() *ListNode {
    return reverseList(l)
}
```

#### **Real-World Applications**

- **Data Structures**: Implement stack using linked list
- **Undo Operations**: Reverse operations in text editors
- **Game Development**: Reverse move sequences
- **Network Protocols**: Reverse packet sequences

### Testing

```go
func TestReverseList(t *testing.T) {
    tests := []struct {
        input    []int
        expected []int
    }{
        {[]int{1, 2, 3, 4, 5}, []int{5, 4, 3, 2, 1}},
        {[]int{1, 2}, []int{2, 1}},
        {[]int{}, []int{}},
        {[]int{1}, []int{1}},
    }

    for _, test := range tests {
        head := buildList(test.input)
        result := reverseList(head)
        actual := listToSlice(result)

        if !reflect.DeepEqual(actual, test.expected) {
            t.Errorf("reverseList(%v) = %v, expected %v",
                test.input, actual, test.expected)
        }
    }
}

// Helper functions
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

func listToSlice(head *ListNode) []int {
    var result []int
    current := head

    for current != nil {
        result = append(result, current.Val)
        current = current.Next
    }

    return result
}
```

### Visualization

```
Original List: 1 -> 2 -> 3 -> 4 -> 5 -> nil

Step 1: prev=nil, current=1, next=2
        1 -> 2 -> 3 -> 4 -> 5 -> nil
        ^
        current

Step 2: prev=1, current=2, next=3
        nil <- 1    2 -> 3 -> 4 -> 5 -> nil
               ^    ^
               prev current

Step 3: prev=2, current=3, next=4
        nil <- 1 <- 2    3 -> 4 -> 5 -> nil
                    ^    ^
                    prev current

Step 4: prev=3, current=4, next=5
        nil <- 1 <- 2 <- 3    4 -> 5 -> nil
                         ^    ^
                         prev current

Step 5: prev=4, current=5, next=nil
        nil <- 1 <- 2 <- 3 <- 4    5 -> nil
                              ^    ^
                              prev current

Final: prev=5, current=nil
       5 -> 4 -> 3 -> 2 -> 1 -> nil
       ^
       new head
```

### Performance Comparison

| Approach  | Time | Space | Pros               | Cons                |
| --------- | ---- | ----- | ------------------ | ------------------- |
| Iterative | O(n) | O(1)  | Space efficient    | More complex logic  |
| Recursive | O(n) | O(n)  | Simple logic       | Stack overflow risk |
| Stack     | O(n) | O(n)  | Easy to understand | Extra space needed  |

**Recommendation**: Use iterative approach for production code, recursive for interviews when space is not a concern.
