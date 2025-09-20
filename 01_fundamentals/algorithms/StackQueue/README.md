# Stack & Queue Pattern

> **Master stack and queue data structures with Go implementations**

## 📋 Problems

### **Stack Problems**

- [Valid Parentheses](ValidParentheses.md/) - Validate bracket sequences
- [Min Stack](MinStack.md/) - Stack with minimum element tracking
- [Daily Temperatures](DailyTemperatures.md/) - Find next warmer temperature
- [Largest Rectangle in Histogram](LargestRectangleInHistogram.md/) - Find largest rectangle
- [Trapping Rain Water](../Arrays/TrappingRainWater.md) - Collect rainwater between bars

### **Queue Problems**

- [Implement Queue using Stacks](ImplementQueueUsingStacks.md/) - Queue implementation
- [Implement Stack using Queues](ImplementStackUsingQueues.md/) - Stack implementation
- [Sliding Window Maximum](../SlidingWindow/SlidingWindowMaximum.md) - Find maximum in sliding window
- [First Unique Character in a String](../Strings/FirstUniqueCharacterInString.md) - Find first unique character
- [Design Circular Queue](DesignCircularQueue.md/) - Circular queue implementation

### **Monotonic Stack/Queue**

- [Next Greater Element](NextGreaterElement.md/) - Find next greater element
- [Next Greater Element II](NextGreaterElementII.md/) - Circular array version
- [Remove K Digits](RemoveKDigits.md/) - Remove digits to get smallest number
- [Largest Rectangle in Histogram](LargestRectangleInHistogram.md/) - Find largest rectangle
- [Trapping Rain Water](../Arrays/TrappingRainWater.md) - Collect rainwater

### **Advanced Stack/Queue**

- [Basic Calculator](BasicCalculator.md/) - Evaluate mathematical expressions
- [Basic Calculator II](BasicCalculatorII.md/) - Calculator with operators
- [Decode String](DecodeString.md/) - Decode encoded strings
- [Flatten Nested List Iterator](FlattenNestedListIterator.md/) - Flatten nested structures
- [Asteroid Collision](AsteroidCollision.md/) - Simulate asteroid collisions

---

## 🎯 Key Concepts

### **Stack Operations**

**Detailed Explanation:**
A stack is a linear data structure that follows the Last In First Out (LIFO) principle. It's one of the most fundamental data structures in computer science, used in many algorithms and real-world applications.

**Core Operations:**

**1. Push Operation:**

- **Definition**: Add an element to the top of the stack
- **Time Complexity**: O(1) amortized
- **Space Complexity**: O(1) per operation
- **Implementation**: Append element to the end of underlying array/slice
- **Use Cases**: Adding new elements, building stacks incrementally
- **Error Handling**: Consider stack overflow in fixed-size implementations

**2. Pop Operation:**

- **Definition**: Remove and return the top element from the stack
- **Time Complexity**: O(1)
- **Space Complexity**: O(1)
- **Implementation**: Remove and return the last element from underlying array/slice
- **Use Cases**: Processing elements in reverse order, backtracking
- **Error Handling**: Check for empty stack before popping

**3. Peek/Top Operation:**

- **Definition**: View the top element without removing it
- **Time Complexity**: O(1)
- **Space Complexity**: O(1)
- **Implementation**: Return the last element without modifying the stack
- **Use Cases**: Checking next element, conditional logic
- **Error Handling**: Return sentinel value or error for empty stack

**4. IsEmpty Operation:**

- **Definition**: Check if the stack contains any elements
- **Time Complexity**: O(1)
- **Space Complexity**: O(1)
- **Implementation**: Check if underlying array/slice length is zero
- **Use Cases**: Loop termination, validation, error prevention
- **Error Handling**: Essential for preventing underflow errors

**Stack Properties:**

- **LIFO Principle**: Last element added is the first to be removed
- **Dynamic Size**: Can grow and shrink as needed (in dynamic implementations)
- **Single Access Point**: Only the top element is accessible
- **Sequential Access**: Elements must be accessed in reverse order of insertion

### **Queue Operations**

**Detailed Explanation:**
A queue is a linear data structure that follows the First In First Out (FIFO) principle. It's essential for many algorithms and real-world scenarios where order matters.

**Core Operations:**

**1. Enqueue Operation:**

- **Definition**: Add an element to the rear (end) of the queue
- **Time Complexity**: O(1) amortized
- **Space Complexity**: O(1) per operation
- **Implementation**: Append element to the end of underlying array/slice
- **Use Cases**: Adding new elements, building queues incrementally
- **Error Handling**: Consider queue overflow in fixed-size implementations

**2. Dequeue Operation:**

- **Definition**: Remove and return the front element from the queue
- **Time Complexity**: O(1) amortized (with proper implementation)
- **Space Complexity**: O(1)
- **Implementation**: Remove and return the first element from underlying array/slice
- **Use Cases**: Processing elements in order, task scheduling
- **Error Handling**: Check for empty queue before dequeuing

**3. Front Operation:**

- **Definition**: View the front element without removing it
- **Time Complexity**: O(1)
- **Space Complexity**: O(1)
- **Implementation**: Return the first element without modifying the queue
- **Use Cases**: Checking next element, conditional logic
- **Error Handling**: Return sentinel value or error for empty queue

**4. IsEmpty Operation:**

- **Definition**: Check if the queue contains any elements
- **Time Complexity**: O(1)
- **Space Complexity**: O(1)
- **Implementation**: Check if underlying array/slice length is zero
- **Use Cases**: Loop termination, validation, error prevention
- **Error Handling**: Essential for preventing underflow errors

**Queue Properties:**

- **FIFO Principle**: First element added is the first to be removed
- **Dynamic Size**: Can grow and shrink as needed (in dynamic implementations)
- **Two Access Points**: Front for removal, rear for insertion
- **Sequential Access**: Elements must be accessed in order of insertion

### **Common Patterns**

**Detailed Explanation:**
Understanding common patterns helps identify when to use stacks and queues, and how to combine them effectively for complex problems.

**1. Monotonic Stack:**

- **Definition**: Stack that maintains elements in increasing or decreasing order
- **Purpose**: Efficiently find next greater/smaller elements
- **Implementation**: Remove elements that violate monotonic property
- **Time Complexity**: O(n) for processing all elements
- **Use Cases**: Next greater element, largest rectangle in histogram, trapping rain water
- **Key Insight**: Elements that are "blocked" by larger elements can be removed

**2. Two Stacks Pattern:**

- **Definition**: Use two stacks to implement complex operations
- **Purpose**: Achieve operations not possible with single stack
- **Implementation**: One stack for input, another for output
- **Time Complexity**: Amortized O(1) for most operations
- **Use Cases**: Queue implementation, expression evaluation, undo operations
- **Key Insight**: Transfer elements between stacks to achieve desired order

**3. Stack with Extra Information:**

- **Definition**: Stack that tracks additional data along with elements
- **Purpose**: Answer queries about stack properties efficiently
- **Implementation**: Store tuples or use parallel data structures
- **Time Complexity**: O(1) for most operations
- **Use Cases**: Min stack, max stack, stack with sum/product
- **Key Insight**: Trade space for time to maintain additional properties

**4. Queue with Two Stacks:**

- **Definition**: Implement queue operations using two stacks
- **Purpose**: Achieve FIFO behavior using LIFO data structures
- **Implementation**: One stack for enqueue, another for dequeue
- **Time Complexity**: Amortized O(1) for most operations
- **Use Cases**: When only stack operations are available, educational purposes
- **Key Insight**: Reverse order by transferring between stacks

**Advanced Patterns:**

- **Deque (Double-ended Queue)**: Supports insertion/deletion at both ends
- **Priority Queue**: Elements have priorities, highest priority served first
- **Circular Queue**: Fixed-size queue with wraparound behavior
- **Stack of Stacks**: Nested stack structure for complex hierarchies
- **Queue of Queues**: Nested queue structure for multi-level processing

**Discussion Questions & Answers:**

**Q1: How do you choose between stack and queue for different problem types in Go?**

**Answer:** Data structure selection criteria:

- **LIFO Requirement**: Use stack for problems requiring last-in-first-out behavior (parentheses matching, expression evaluation, backtracking)
- **FIFO Requirement**: Use queue for problems requiring first-in-first-out behavior (BFS traversal, task scheduling, buffering)
- **Nested Structures**: Use stack for nested or hierarchical structures (parentheses, brackets, nested lists)
- **Order Preservation**: Use queue when order of processing matters (BFS, level-order traversal)
- **Backtracking**: Use stack for backtracking algorithms (DFS, recursive problems)
- **Sliding Window**: Use deque for sliding window problems with efficient min/max operations
- **Expression Evaluation**: Use stack for postfix/infix expression evaluation
- **Monotonic Properties**: Use monotonic stack for next greater/smaller element problems
- **Memory Constraints**: Consider space complexity and choose accordingly
- **Performance Requirements**: Consider time complexity and choose the most efficient option

**Q2: What are the common pitfalls when implementing stacks and queues in Go?**

**Answer:** Common implementation pitfalls:

- **Slice Operations**: Not handling slice bounds correctly, especially in pop/dequeue operations
- **Memory Management**: Not properly managing slice capacity, leading to unnecessary allocations
- **Error Handling**: Not checking for empty stack/queue before pop/dequeue operations
- **Index Management**: Off-by-one errors in array-based implementations
- **Concurrency**: Not handling concurrent access properly in multi-threaded environments
- **Type Safety**: Issues with generic implementations and type assertions
- **Performance**: Using inefficient operations (O(n) dequeue in naive implementations)
- **Edge Cases**: Not handling empty data structures, single elements, or maximum capacity
- **Testing**: Not testing with various data sizes and edge cases
- **Documentation**: Not documenting time/space complexity and usage patterns

**Q3: How do you optimize stack and queue implementations for performance in Go?**

**Answer:** Performance optimization strategies:

- **Pre-allocation**: Use make() with known capacity to avoid repeated allocations
- **Slice Management**: Use slice tricks for efficient operations (append, slicing)
- **Memory Pooling**: Reuse data structures to reduce garbage collection pressure
- **Efficient Dequeue**: Use circular buffer or two-stack approach for O(1) dequeue
- **Monotonic Stack**: Implement efficiently by removing unnecessary elements
- **Batch Operations**: Process multiple elements at once when possible
- **Type Optimization**: Use appropriate data types (int vs int64) based on requirements
- **Cache Locality**: Access elements sequentially when possible
- **Avoid Unnecessary Copies**: Use pointers or references when appropriate
- **Profiling**: Use Go profiling tools to identify performance bottlenecks
- **Benchmarking**: Write benchmarks to measure and compare performance
- **Memory Layout**: Consider memory layout for better cache performance

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
