# Two Pointers Pattern

> **Master two pointers technique for array and string problems with Go implementations**

## 📋 Problems

### **Array Two Pointers**
- [Two Sum](./TwoSum.md) - Find two numbers that sum to target
- [3Sum](./3Sum.md) - Find three numbers that sum to zero
- [4Sum](./4Sum.md) - Find four numbers that sum to target
- [Container With Most Water](./ContainerWithMostWater.md) - Maximum area between lines
- [Trapping Rain Water](./TrappingRainWater.md) - Collect rainwater between bars

### **String Two Pointers**
- [Valid Palindrome](./ValidPalindrome.md) - Check if string is palindrome
- [Valid Palindrome II](./ValidPalindromeII.md) - Palindrome with one deletion
- [Reverse String](./ReverseString.md) - Reverse string in-place
- [Remove Duplicates](./RemoveDuplicates.md) - Remove duplicates from sorted array
- [Move Zeroes](./MoveZeroes.md) - Move zeros to end

### **Fast and Slow Pointers**
- [Linked List Cycle](./LinkedListCycle.md) - Detect cycle in linked list
- [Linked List Cycle II](./LinkedListCycleII.md) - Find cycle start node
- [Middle of the Linked List](./MiddleOfLinkedList.md) - Find middle node
- [Remove Nth Node From End](./RemoveNthNodeFromEnd.md) - Remove nth node from end
- [Palindrome Linked List](./PalindromeLinkedList.md) - Check if list is palindrome

---

## 🎯 Key Concepts

### **Two Pointers Types**
1. **Opposite Direction**: Start from both ends, move towards center
2. **Same Direction**: Both pointers move in same direction
3. **Fast and Slow**: One pointer moves faster than the other

### **When to Use Two Pointers**
- **Sorted Arrays**: Find pairs, triplets, or quadruplets
- **Palindrome Problems**: Check symmetry from both ends
- **Cycle Detection**: Use fast and slow pointers
- **Window Problems**: Maintain a window with two pointers

### **Common Patterns**
- **Sum Problems**: Find elements with specific sum
- **Palindrome**: Check symmetry
- **Cycle Detection**: Floyd's algorithm
- **Window Sliding**: Maintain valid window

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
