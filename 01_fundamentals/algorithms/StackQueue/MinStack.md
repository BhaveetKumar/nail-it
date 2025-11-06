---
# Auto-generated front matter
Title: Minstack
LastUpdated: 2025-11-06T20:45:58.705014
Tags: []
Status: draft
---

# Min Stack

### Problem
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:
- `MinStack()` initializes the stack object.
- `void push(int val)` pushes the element val onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

**Example:**
```
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]
```

### Golang Solution

```go
type MinStack struct {
    stack    []int
    minStack []int
}

func Constructor() MinStack {
    return MinStack{
        stack:    make([]int, 0),
        minStack: make([]int, 0),
    }
}

func (ms *MinStack) Push(val int) {
    ms.stack = append(ms.stack, val)
    
    if len(ms.minStack) == 0 || val <= ms.minStack[len(ms.minStack)-1] {
        ms.minStack = append(ms.minStack, val)
    }
}

func (ms *MinStack) Pop() {
    if len(ms.stack) == 0 {
        return
    }
    
    val := ms.stack[len(ms.stack)-1]
    ms.stack = ms.stack[:len(ms.stack)-1]
    
    if len(ms.minStack) > 0 && val == ms.minStack[len(ms.minStack)-1] {
        ms.minStack = ms.minStack[:len(ms.minStack)-1]
    }
}

func (ms *MinStack) Top() int {
    return ms.stack[len(ms.stack)-1]
}

func (ms *MinStack) GetMin() int {
    return ms.minStack[len(ms.minStack)-1]
}
```

### Complexity
- **Time Complexity:** O(1) for all operations
- **Space Complexity:** O(n)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
