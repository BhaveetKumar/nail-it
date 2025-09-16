# Sliding Window Pattern

> **Master sliding window techniques for array and string problems with Go implementations**

## 📋 Problems

### **Fixed Window Size**

- [Maximum Sum Subarray of Size K](MaximumSumSubarrayOfSizeK.md/) - Find maximum sum of fixed window
- [Average of Subarrays of Size K](AverageOfSubarraysOfSizeK.md/) - Calculate averages
- [First Negative Number in Every Window](FirstNegativeNumberInEveryWindow.md/) - Find first negative in each window
- [Count Anagrams](CountAnagrams.md/) - Count anagrams in string
- [Maximum of All Subarrays of Size K](MaximumOfAllSubarraysOfSizeK.md/) - Find maximum in each window

### **Variable Window Size**

- [Longest Substring Without Repeating Characters](LongestSubstringWithoutRepeatingCharacters.md/) - Find longest unique substring
- [Minimum Window Substring](MinimumWindowSubstring.md/) - Find minimum window containing all characters
- [Longest Substring with At Most K Distinct Characters](LongestSubstringWithAtMostKDistinctCharacters.md/) - Find longest substring with K distinct chars
- [Longest Repeating Character Replacement](LongestRepeatingCharacterReplacement.md/) - Replace characters to get longest substring
- [Fruit Into Baskets](FruitIntoBaskets.md/) - Collect maximum fruits

### **Advanced Sliding Window**

- [Subarray Sum Equals K](SubarraySumEqualsK.md/) - Find subarrays with sum K
- [Permutation in String](PermutationInString.md/) - Check if permutation exists
- [Find All Anagrams in a String](FindAllAnagramsInString.md/) - Find all anagram occurrences
- [Sliding Window Maximum](SlidingWindowMaximum.md/) - Find maximum in sliding window
- [Longest Subarray with Sum at Most K](LongestSubarrayWithSumAtMostK.md/) - Find longest subarray with sum ≤ K

---

## 🎯 Key Concepts

### **Sliding Window Types**

**Detailed Explanation:**
The sliding window technique is a powerful algorithmic pattern that efficiently solves problems involving contiguous subarrays or substrings. It maintains a "window" of elements and slides it across the data structure, avoiding redundant calculations by reusing previous computations.

**1. Fixed Window:**

- **Definition**: Window size remains constant throughout the algorithm
- **Characteristics**: Window size is predetermined and doesn't change
- **Time Complexity**: O(n) - single pass through the array
- **Space Complexity**: O(1) - constant extra space
- **Use Cases**: Maximum sum of subarray of size k, average of subarrays
- **Algorithm**: Calculate first window, then slide by removing leftmost and adding rightmost element
- **Example**: Find maximum sum of subarray of size 3 in [1,2,3,4,5] → windows: [1,2,3], [2,3,4], [3,4,5]

**2. Variable Window:**

- **Definition**: Window size changes based on certain conditions
- **Characteristics**: Window expands and contracts dynamically
- **Time Complexity**: O(n) - each element is added and removed at most once
- **Space Complexity**: O(k) where k is the number of distinct elements
- **Use Cases**: Longest substring without repeating characters, minimum window substring
- **Algorithm**: Expand window until condition is met, then contract until condition is violated
- **Example**: Find longest substring with at most 2 distinct characters

**3. Two Pointers:**

- **Definition**: Use left and right pointers to define window boundaries
- **Characteristics**: Both pointers can move independently
- **Time Complexity**: O(n) - each pointer moves at most n times
- **Space Complexity**: O(1) - constant extra space
- **Use Cases**: Two sum in sorted array, container with most water
- **Algorithm**: Move pointers based on comparison with target or condition
- **Example**: Find two numbers that sum to target in sorted array

**Advanced Window Types:**

- **Deque-based Window**: Use deque to maintain window with efficient min/max operations
- **Hash-based Window**: Use hash map to track frequency of elements in window
- **Prefix Sum Window**: Use prefix sums for efficient range sum calculations
- **Monotonic Window**: Maintain monotonic property in window (increasing/decreasing)

### **When to Use Sliding Window**

**Detailed Explanation:**
The sliding window technique is most effective when dealing with problems that involve contiguous elements and can benefit from avoiding redundant calculations.

**Problem Characteristics:**

**1. Subarray/Substring Problems:**

- **Nature**: Problems involving contiguous elements in arrays or strings
- **Examples**: Maximum sum subarray, longest substring, minimum window substring
- **Strategy**: Maintain a window of elements and slide it across the data
- **Benefit**: Avoid recalculating overlapping portions of the window

**2. Contiguous Elements:**

- **Nature**: Problems that require working with consecutive elements
- **Examples**: Subarray problems, substring problems, window-based calculations
- **Strategy**: Use two pointers to define window boundaries
- **Benefit**: Efficiently process all possible contiguous subarrays/substrings

**3. Optimization Problems:**

- **Nature**: Problems that require finding optimal (maximum/minimum) solutions
- **Examples**: Maximum sum, longest length, minimum size
- **Strategy**: Track optimal solution while sliding the window
- **Benefit**: Find optimal solution in linear time

**4. Frequency Problems:**

- **Nature**: Problems involving counting elements or characters in a window
- **Examples**: Anagram detection, character frequency, element counting
- **Strategy**: Use hash map to track frequency of elements in window
- **Benefit**: Efficiently maintain frequency counts as window slides

**When NOT to Use Sliding Window:**

- **Non-contiguous Elements**: When elements don't need to be contiguous
- **Non-sequential Processing**: When order of processing doesn't matter
- **Complex Conditions**: When window validity conditions are too complex
- **Sparse Data**: When most windows are invalid or empty

### **Common Patterns**

**Detailed Explanation:**
Sliding window algorithms follow specific patterns that can be recognized and applied to solve similar problems efficiently.

**1. Expand Right:**

- **Purpose**: Increase window size by moving right pointer
- **When to Use**: When current window is valid and we want to explore larger windows
- **Implementation**: Increment right pointer and update window state
- **Example**: In longest substring problem, expand to find longer valid substrings
- **Time Complexity**: O(1) per expansion

**2. Contract Left:**

- **Purpose**: Decrease window size by moving left pointer
- **When to Use**: When current window violates the condition
- **Implementation**: Increment left pointer and update window state
- **Example**: In minimum window problem, contract to find smaller valid windows
- **Time Complexity**: O(1) per contraction

**3. Maintain Invariant:**

- **Purpose**: Keep window in a valid state according to problem constraints
- **When to Use**: Throughout the algorithm to ensure correctness
- **Implementation**: Check and update window state after each pointer movement
- **Example**: Maintain frequency count of characters in window
- **Time Complexity**: O(1) per maintenance operation

**4. Update Result:**

- **Purpose**: Track the optimal solution found so far
- **When to Use**: After each valid window is found
- **Implementation**: Compare current window with best solution and update if better
- **Example**: Update maximum length or minimum size when valid window is found
- **Time Complexity**: O(1) per update

**Advanced Patterns:**

- **Deque Pattern**: Use deque to maintain window with efficient min/max operations
- **Hash Map Pattern**: Use hash map to track frequency or count of elements
- **Prefix Sum Pattern**: Use prefix sums for efficient range sum calculations
- **Monotonic Pattern**: Maintain monotonic property in window elements

**Discussion Questions & Answers:**

**Q1: How do you optimize sliding window algorithms for performance in Go?**

**Answer:** Performance optimization strategies:

- **Efficient Data Structures**: Use arrays instead of maps when possible (for small character sets)
- **Memory Management**: Reuse variables and avoid unnecessary allocations
- **Early Termination**: Stop processing when optimal solution is found
- **Incremental Updates**: Update window state incrementally instead of recalculating
- **Pointer Management**: Use efficient pointer arithmetic and avoid bounds checking
- **Cache Locality**: Access elements in sequential order for better cache performance
- **String Optimization**: Use byte arrays for string operations when possible
- **Profiling**: Use Go profiling tools to identify performance bottlenecks
- **Algorithm Selection**: Choose the most efficient sliding window variant for the problem
- **Memory Pooling**: Reuse data structures to reduce garbage collection pressure

**Q2: What are the common pitfalls when implementing sliding window algorithms in Go?**

**Answer:** Common pitfalls include:

- **Off-by-One Errors**: Incorrect window size calculations or boundary conditions
- **Pointer Management**: Not properly updating left and right pointers
- **State Management**: Incorrectly maintaining window state (frequency counts, sums)
- **Edge Cases**: Not handling empty arrays, single elements, or invalid inputs
- **Memory Leaks**: Not properly cleaning up hash maps or other data structures
- **Index Bounds**: Accessing array elements outside valid bounds
- **String Handling**: Issues with Unicode strings and rune handling
- **Type Conversions**: Problems with type conversions between different numeric types
- **Infinite Loops**: Not properly terminating the sliding window loop
- **State Synchronization**: Not keeping window state consistent with pointer positions

**Q3: How do you handle complex sliding window problems with multiple constraints?**

**Answer:** Complex sliding window problem handling:

- **Multiple Conditions**: Use multiple data structures to track different constraints
- **State Machine**: Implement state machine to handle complex window validity
- **Layered Approach**: Break down complex problems into simpler sliding window subproblems
- **Hash Map Management**: Use multiple hash maps to track different types of elements
- **Constraint Validation**: Implement efficient constraint checking functions
- **Window State**: Maintain comprehensive window state with all required information
- **Backtracking**: Use backtracking when simple sliding window is not sufficient
- **Optimization**: Use mathematical insights to optimize constraint checking
- **Testing**: Implement comprehensive tests for all constraint combinations
- **Documentation**: Clearly document all constraints and their interactions

---

## 🛠️ Go-Specific Tips

### **Fixed Window Template**

```go
func maxSumSubarray(nums []int, k int) int {
    if len(nums) < k {
        return 0
    }

    // Calculate sum of first window
    windowSum := 0
    for i := 0; i < k; i++ {
        windowSum += nums[i]
    }

    maxSum := windowSum

    // Slide the window
    for i := k; i < len(nums); i++ {
        windowSum = windowSum - nums[i-k] + nums[i]
        maxSum = max(maxSum, windowSum)
    }

    return maxSum
}
```

### **Variable Window Template**

```go
func longestSubstring(s string, k int) int {
    left := 0
    maxLen := 0
    charCount := make(map[byte]int)

    for right := 0; right < len(s); right++ {
        // Expand window
        charCount[s[right]]++

        // Contract window while maintaining condition
        for len(charCount) > k {
            charCount[s[left]]--
            if charCount[s[left]] == 0 {
                delete(charCount, s[left])
            }
            left++
        }

        // Update result
        maxLen = max(maxLen, right-left+1)
    }

    return maxLen
}
```

### **Two Pointers Technique**

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

### **Frequency Map with Sliding Window**

```go
func minWindow(s string, t string) string {
    if len(s) < len(t) {
        return ""
    }

    // Count characters in target string
    targetCount := make(map[byte]int)
    for i := 0; i < len(t); i++ {
        targetCount[t[i]]++
    }

    left := 0
    minLen := len(s) + 1
    minStart := 0
    matched := 0

    for right := 0; right < len(s); right++ {
        // Expand window
        if targetCount[s[right]] > 0 {
            matched++
        }
        targetCount[s[right]]--

        // Contract window
        for matched == len(t) {
            if right-left+1 < minLen {
                minLen = right - left + 1
                minStart = left
            }

            targetCount[s[left]]++
            if targetCount[s[left]] > 0 {
                matched--
            }
            left++
        }
    }

    if minLen > len(s) {
        return ""
    }

    return s[minStart : minStart+minLen]
}
```

---

## 🎯 Interview Tips

### **How to Identify Sliding Window Problems**

1. **Subarray/Substring**: Find optimal contiguous elements
2. **Window Size**: Fixed or variable window size
3. **Optimization**: Find maximum/minimum in window
4. **Frequency**: Count elements in window

### **Common Sliding Window Problem Patterns**

- **Maximum Sum**: Find maximum sum in window
- **Longest Substring**: Find longest substring with condition
- **Minimum Window**: Find minimum window with condition
- **Anagram Detection**: Find anagrams in string
- **Two Pointers**: Use pointers to define window

### **Optimization Tips**

- **Pre-calculate**: Calculate first window separately
- **Incremental Updates**: Update window incrementally
- **Early Termination**: Stop when condition is met
- **Space Optimization**: Use arrays instead of maps when possible
