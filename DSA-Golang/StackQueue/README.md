# Stack & Queue Pattern

> **Master stack and queue data structures with Go implementations**

## 📋 Problems

### **Stack Problems**
- [Valid Parentheses](./ValidParentheses.md) - Validate bracket sequences
- [Min Stack](./MinStack.md) - Stack with minimum element tracking
- [Daily Temperatures](./DailyTemperatures.md) - Find next warmer temperature
- [Largest Rectangle in Histogram](./LargestRectangleInHistogram.md) - Find largest rectangle
- [Trapping Rain Water](./TrappingRainWater.md) - Collect rainwater between bars

### **Queue Problems**
- [Implement Queue using Stacks](./ImplementQueueUsingStacks.md) - Queue implementation
- [Implement Stack using Queues](./ImplementStackUsingQueues.md) - Stack implementation
- [Sliding Window Maximum](./SlidingWindowMaximum.md) - Find maximum in sliding window
- [First Unique Character in a String](./FirstUniqueCharacterInString.md) - Find first unique character
- [Design Circular Queue](./DesignCircularQueue.md) - Circular queue implementation

### **Monotonic Stack/Queue**
- [Next Greater Element](./NextGreaterElement.md) - Find next greater element
- [Next Greater Element II](./NextGreaterElementII.md) - Circular array version
- [Remove K Digits](./RemoveKDigits.md) - Remove digits to get smallest number
- [Largest Rectangle in Histogram](./LargestRectangleInHistogram.md) - Find largest rectangle
- [Trapping Rain Water](./TrappingRainWater.md) - Collect rainwater

### **Advanced Stack/Queue**
- [Basic Calculator](./BasicCalculator.md) - Evaluate mathematical expressions
- [Basic Calculator II](./BasicCalculatorII.md) - Calculator with operators
- [Decode String](./DecodeString.md) - Decode encoded strings
- [Flatten Nested List Iterator](./FlattenNestedListIterator.md) - Flatten nested structures
- [Asteroid Collision](./AsteroidCollision.md) - Simulate asteroid collisions

---

## 🎯 Key Concepts

### **Stack Operations**
- **Push**: Add element to top
- **Pop**: Remove element from top
- **Peek/Top**: View top element without removing
- **IsEmpty**: Check if stack is empty

### **Queue Operations**
- **Enqueue**: Add element to rear
- **Dequeue**: Remove element from front
- **Front**: View front element without removing
- **IsEmpty**: Check if queue is empty

### **Common Patterns**
- **Monotonic Stack**: Maintain increasing/decreasing order
- **Two Stacks**: Use two stacks for complex operations
- **Stack with Extra Info**: Track additional information
- **Queue with Two Stacks**: Implement queue using stacks

---

## 🛠️ Go-Specific Tips

### **Stack Implementation**
```go
type Stack struct {
    items []int
}

func NewStack() *Stack {
    return &Stack{items: make([]int, 0)}
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() int {
    if len(s.items) == 0 {
        return -1 // or handle error
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item
}

func (s *Stack) Peek() int {
    if len(s.items) == 0 {
        return -1
    }
    return s.items[len(s.items)-1]
}

func (s *Stack) IsEmpty() bool {
    return len(s.items) == 0
}
```

### **Queue Implementation**
```go
type Queue struct {
    items []int
}

func NewQueue() *Queue {
    return &Queue{items: make([]int, 0)}
}

func (q *Queue) Enqueue(item int) {
    q.items = append(q.items, item)
}

func (q *Queue) Dequeue() int {
    if len(q.items) == 0 {
        return -1
    }
    item := q.items[0]
    q.items = q.items[1:]
    return item
}

func (q *Queue) Front() int {
    if len(q.items) == 0 {
        return -1
    }
    return q.items[0]
}

func (q *Queue) IsEmpty() bool {
    return len(q.items) == 0
}
```

### **Monotonic Stack**
```go
func nextGreaterElement(nums []int) []int {
    result := make([]int, len(nums))
    stack := make([]int, 0)
    
    for i := len(nums) - 1; i >= 0; i-- {
        // Remove elements smaller than current
        for len(stack) > 0 && stack[len(stack)-1] <= nums[i] {
            stack = stack[:len(stack)-1]
        }
        
        if len(stack) > 0 {
            result[i] = stack[len(stack)-1]
        } else {
            result[i] = -1
        }
        
        stack = append(stack, nums[i])
    }
    
    return result
}
```

### **Queue with Two Stacks**
```go
type MyQueue struct {
    input  []int
    output []int
}

func Constructor() MyQueue {
    return MyQueue{
        input:  make([]int, 0),
        output: make([]int, 0),
    }
}

func (q *MyQueue) Push(x int) {
    q.input = append(q.input, x)
}

func (q *MyQueue) Pop() int {
    q.moveInputToOutput()
    if len(q.output) == 0 {
        return -1
    }
    item := q.output[len(q.output)-1]
    q.output = q.output[:len(q.output)-1]
    return item
}

func (q *MyQueue) Peek() int {
    q.moveInputToOutput()
    if len(q.output) == 0 {
        return -1
    }
    return q.output[len(q.output)-1]
}

func (q *MyQueue) Empty() bool {
    return len(q.input) == 0 && len(q.output) == 0
}

func (q *MyQueue) moveInputToOutput() {
    if len(q.output) == 0 {
        for len(q.input) > 0 {
            q.output = append(q.output, q.input[len(q.input)-1])
            q.input = q.input[:len(q.input)-1]
        }
    }
}
```

---

## 🎯 Interview Tips

### **How to Identify Stack/Queue Problems**
1. **LIFO/FIFO**: Last In First Out or First In First Out
2. **Nested Structures**: Parentheses, brackets, nested lists
3. **Next Greater/Smaller**: Find next element with property
4. **Expression Evaluation**: Mathematical expressions
5. **Sliding Window**: Use deque for sliding window problems

### **Common Stack/Queue Problem Patterns**
- **Parentheses Matching**: Use stack for nested structures
- **Monotonic Stack**: Maintain increasing/decreasing order
- **Expression Evaluation**: Use stack for postfix/infix
- **Sliding Window**: Use deque for efficient operations
- **Two Stacks**: Use two stacks for complex operations

### **Optimization Tips**
- **Pre-allocate**: Use make() with capacity when size is known
- **Monotonic Stack**: Use for next greater/smaller problems
- **Deque**: Use for sliding window maximum/minimum
- **Stack with Extra Info**: Track additional information
