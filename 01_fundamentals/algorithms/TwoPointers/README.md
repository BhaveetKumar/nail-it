# Two Pointers Pattern

> **Master two pointers technique for array and string problems with Go implementations**

## 📋 Problems

### **Array Two Pointers**

- [Two Sum](TwoSum.md/) - Find two numbers that sum to target
- [3Sum](3Sum.md/) - Find three numbers that sum to zero
- [4Sum](4Sum.md/) - Find four numbers that sum to target
- [Container With Most Water](ContainerWithMostWater.md/) - Maximum area between lines
- [Trapping Rain Water](TrappingRainWater.md/) - Collect rainwater between bars

### **String Two Pointers**

- [Valid Palindrome](ValidPalindrome.md/) - Check if string is palindrome
- [Valid Palindrome II](ValidPalindromeII.md/) - Palindrome with one deletion
- [Reverse String](ReverseString.md/) - Reverse string in-place
- [Remove Duplicates](RemoveDuplicates.md/) - Remove duplicates from sorted array
- [Move Zeroes](MoveZeroes.md/) - Move zeros to end

### **Fast and Slow Pointers**

- [Linked List Cycle](LinkedListCycle.md/) - Detect cycle in linked list
- [Linked List Cycle II](LinkedListCycleII.md/) - Find cycle start node
- [Middle of the Linked List](MiddleOfLinkedList.md/) - Find middle node
- [Remove Nth Node From End](RemoveNthNodeFromEnd.md/) - Remove nth node from end
- [Palindrome Linked List](PalindromeLinkedList.md/) - Check if list is palindrome

---

## 🎯 Key Concepts

### **Two Pointers Types**

**Detailed Explanation:**
The two pointers technique is a fundamental algorithmic pattern that uses two pointers to traverse data structures efficiently. It's particularly powerful for solving problems that involve searching, comparing, or manipulating elements in arrays, strings, or linked lists.

**1. Opposite Direction Pointers:**

- **Definition**: Two pointers start from opposite ends and move towards each other
- **Movement**: One pointer moves from left to right, another from right to left
- **Use Cases**: Sorted arrays, palindrome checking, finding pairs with specific properties
- **Time Complexity**: O(n) for most problems
- **Space Complexity**: O(1) - no extra space needed
- **Key Insight**: Leverage sorted order to eliminate impossible combinations
- **Example**: Finding two numbers that sum to a target in a sorted array

**2. Same Direction Pointers:**

- **Definition**: Both pointers start from the same end and move in the same direction
- **Movement**: Both pointers move from left to right (or right to left)
- **Use Cases**: Removing duplicates, partitioning arrays, sliding window problems
- **Time Complexity**: O(n) for most problems
- **Space Complexity**: O(1) - in-place operations
- **Key Insight**: One pointer tracks position for writing, another for reading
- **Example**: Removing duplicates from sorted array in-place

**3. Fast and Slow Pointers:**

- **Definition**: One pointer moves faster than the other (typically 2x speed)
- **Movement**: Slow pointer moves one step, fast pointer moves two steps
- **Use Cases**: Cycle detection, finding middle element, detecting patterns
- **Time Complexity**: O(n) for most problems
- **Space Complexity**: O(1) - no extra space needed
- **Key Insight**: Different speeds help detect cycles and find specific positions
- **Example**: Detecting cycle in linked list using Floyd's algorithm

**Advanced Pointer Patterns:**

- **Three Pointers**: Extend two pointers to three for more complex problems
- **Multiple Fast/Slow**: Use multiple pointer pairs for complex traversals
- **Bidirectional**: Pointers can move in both directions based on conditions
- **Conditional Movement**: Pointers move based on specific conditions or comparisons

### **When to Use Two Pointers**

**Detailed Explanation:**
Understanding when to apply the two pointers technique is crucial for efficient problem solving. The technique is particularly effective for problems that can be solved by maintaining two positions in a data structure.

**1. Sorted Arrays:**

- **Why**: Sorted order allows us to make intelligent decisions about which elements to consider
- **Problems**: Finding pairs, triplets, or quadruplets with specific properties
- **Strategy**: Use opposite direction pointers to eliminate impossible combinations
- **Examples**: Two Sum, 3Sum, 4Sum, Container With Most Water
- **Key Insight**: If current sum is too small, move left pointer right; if too large, move right pointer left

**2. Palindrome Problems:**

- **Why**: Palindromes have symmetric properties that can be checked from both ends
- **Problems**: Validating palindromes, finding palindromic substrings
- **Strategy**: Use opposite direction pointers to compare characters from both ends
- **Examples**: Valid Palindrome, Palindrome Linked List, Longest Palindromic Substring
- **Key Insight**: Compare characters from both ends and move inward

**3. Cycle Detection:**

- **Why**: Different pointer speeds help detect cycles in linked structures
- **Problems**: Detecting cycles, finding cycle start, detecting patterns
- **Strategy**: Use fast and slow pointers with different movement speeds
- **Examples**: Linked List Cycle, Linked List Cycle II, Find Duplicate Number
- **Key Insight**: If there's a cycle, fast pointer will eventually catch up to slow pointer

**4. Window Problems:**

- **Why**: Two pointers can maintain a valid window efficiently
- **Problems**: Sliding window, substring problems, subarray problems
- **Strategy**: Use same direction pointers to maintain window boundaries
- **Examples**: Longest Substring Without Repeating Characters, Minimum Window Substring
- **Key Insight**: Expand window when valid, contract when invalid

**5. Array Manipulation:**

- **Why**: Two pointers can efficiently partition or transform arrays
- **Problems**: Removing elements, partitioning, reordering
- **Strategy**: Use same direction pointers for in-place operations
- **Examples**: Remove Duplicates, Move Zeroes, Partition Array
- **Key Insight**: One pointer for reading, another for writing

### **Common Patterns**

**Detailed Explanation:**
Recognizing common patterns helps identify when to use two pointers and how to implement them effectively.

**1. Sum Problems:**

- **Pattern**: Find elements that sum to a specific target
- **Strategy**: Use opposite direction pointers on sorted arrays
- **Optimization**: Skip duplicates to avoid duplicate results
- **Examples**: Two Sum, 3Sum, 4Sum, Triplet Sum
- **Key Insight**: Sort array first, then use two pointers to find pairs

**2. Palindrome Pattern:**

- **Pattern**: Check if sequence is symmetric
- **Strategy**: Use opposite direction pointers to compare from both ends
- **Optimization**: Stop early when asymmetry is found
- **Examples**: Valid Palindrome, Palindrome Linked List, Longest Palindromic Substring
- **Key Insight**: Compare characters from both ends and move inward

**3. Cycle Detection Pattern:**

- **Pattern**: Detect cycles in linked structures
- **Strategy**: Use fast and slow pointers with different speeds
- **Optimization**: Use Floyd's algorithm for optimal cycle detection
- **Examples**: Linked List Cycle, Linked List Cycle II, Find Duplicate Number
- **Key Insight**: Fast pointer will catch slow pointer if cycle exists

**4. Window Sliding Pattern:**

- **Pattern**: Maintain a valid window with two pointers
- **Strategy**: Use same direction pointers to expand/contract window
- **Optimization**: Use hash map for efficient window validation
- **Examples**: Longest Substring Without Repeating Characters, Minimum Window Substring
- **Key Insight**: Expand window when valid, contract when invalid

**5. Array Manipulation Pattern:**

- **Pattern**: Transform array in-place using two pointers
- **Strategy**: Use same direction pointers for reading and writing
- **Optimization**: Minimize unnecessary operations
- **Examples**: Remove Duplicates, Move Zeroes, Partition Array
- **Key Insight**: One pointer for reading, another for writing

**Discussion Questions & Answers:**

**Q1: How do you choose between different two pointers techniques for different problem types in Go?**

**Answer:** Two pointers technique selection:

- **Opposite Direction**: Use for sorted arrays, palindrome problems, and problems where you can eliminate possibilities
- **Same Direction**: Use for array manipulation, sliding window, and in-place operations
- **Fast and Slow**: Use for cycle detection, finding middle elements, and pattern detection
- **Problem Analysis**: Analyze the problem to determine which pointer movement makes sense
- **Data Structure**: Consider the data structure (array, string, linked list) and its properties
- **Optimization**: Choose the technique that provides the best time/space complexity
- **Implementation**: Consider ease of implementation and potential for errors
- **Edge Cases**: Think about edge cases and how different techniques handle them
- **Memory Usage**: Consider space complexity requirements
- **Performance**: Consider performance requirements and constraints

**Q2: What are the common pitfalls when implementing two pointers algorithms in Go?**

**Answer:** Common implementation pitfalls:

- **Index Bounds**: Not handling array bounds correctly, especially in opposite direction pointers
- **Pointer Movement**: Incorrect pointer movement logic leading to infinite loops or missed elements
- **Edge Cases**: Not handling empty arrays, single elements, or arrays with all same elements
- **Off-by-One Errors**: Common in array indexing and pointer movement
- **Cycle Detection**: Incorrect implementation of Floyd's algorithm for cycle detection
- **Duplicate Handling**: Not properly handling duplicates in sum problems
- **Memory Management**: Not properly managing slice operations and memory
- **Type Safety**: Issues with type conversions and generic implementations
- **Testing**: Not testing with various input sizes and edge cases
- **Documentation**: Not documenting the algorithm logic and complexity

**Q3: How do you optimize two pointers algorithms for performance and memory usage in Go?**

**Answer:** Performance optimization strategies:

- **Pre-allocation**: Use make() with known capacity to avoid repeated allocations
- **In-place Operations**: Use in-place operations when possible to save memory
- **Early Termination**: Stop early when conditions are met to save time
- **Skip Duplicates**: Efficiently skip duplicates in sum problems
- **Efficient Comparisons**: Use efficient comparison operations
- **Memory Layout**: Consider memory layout for better cache performance
- **Slice Operations**: Use efficient slice operations and avoid unnecessary copies
- **Type Optimization**: Use appropriate data types based on requirements
- **Algorithm Selection**: Choose the most efficient algorithm for the problem
- **Profiling**: Use Go profiling tools to identify performance bottlenecks
- **Benchmarking**: Write benchmarks to measure and compare performance
- **Code Review**: Review code for potential optimizations and improvements

---

## 🛠️ Go-Specific Tips

### **Opposite Direction Pointers**

```go
func twoSum(nums []int, target int) []int {
    left, right := 0, len(nums)-1

    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{left, right}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }

    return nil
}
```

### **Same Direction Pointers**

```go
func removeDuplicates(nums []int) int {
    if len(nums) == 0 {
        return 0
    }

    slow := 0
    for fast := 1; fast < len(nums); fast++ {
        if nums[fast] != nums[slow] {
            slow++
            nums[slow] = nums[fast]
        }
    }

    return slow + 1
}
```

### **Fast and Slow Pointers**

```go
func hasCycle(head *ListNode) bool {
    if head == nil || head.Next == nil {
        return false
    }

    slow := head
    fast := head.Next

    for fast != nil && fast.Next != nil {
        if slow == fast {
            return true
        }
        slow = slow.Next
        fast = fast.Next.Next
    }

    return false
}
```

---

## 🎯 Interview Tips

### **How to Identify Two Pointers Problems**

1. **Sorted Array**: Look for sum, pair, or triplet problems
2. **Palindrome**: Check symmetry from both ends
3. **Cycle**: Detect cycles in linked lists
4. **Window**: Maintain sliding window

### **Common Two Pointers Problem Patterns**

- **Sum Problems**: Two Sum, 3Sum, 4Sum
- **Palindrome**: Valid Palindrome, Palindrome Linked List
- **Cycle Detection**: Floyd's cycle detection
- **Array Manipulation**: Remove duplicates, move elements

### **Optimization Tips**

- **Sort First**: Sort array for opposite direction pointers
- **Skip Duplicates**: Handle duplicates in sum problems
- **Early Termination**: Stop when condition is met
- **Memory Efficiency**: Use O(1) space when possible
