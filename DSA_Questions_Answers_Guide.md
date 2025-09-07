# 🧮 Data Structures & Algorithms - Complete Guide

> **Comprehensive DSA guide with Go implementations, optimal solutions, and FAANG interview questions**

## 📋 Table of Contents

1. [Array & String Problems](#array--string-problems)
2. [Linked List Problems](#linked-list-problems)
3. [Tree & Graph Problems](#tree--graph-problems)
4. [Dynamic Programming](#dynamic-programming)
5. [Sorting & Searching](#sorting--searching)
6. [Greedy Algorithms](#greedy-algorithms)
7. [Backtracking](#backtracking)
8. [FAANG Interview Questions](#faang-interview-questions)

---

## 📊 Array & String Problems

### **1. Two Sum**
**Problem**: Given an array of integers and a target sum, find two numbers that add up to the target.

```go
package main

import "fmt"

// Time: O(n), Space: O(n)
func twoSum(nums []int, target int) []int {
    numMap := make(map[int]int)
    
    for i, num := range nums {
        complement := target - num
        if index, exists := numMap[complement]; exists {
            return []int{index, i}
        }
        numMap[num] = i
    }
    
    return []int{}
}

// Time: O(n²), Space: O(1)
func twoSumBruteForce(nums []int, target int) []int {
    for i := 0; i < len(nums); i++ {
        for j := i + 1; j < len(nums); j++ {
            if nums[i] + nums[j] == target {
                return []int{i, j}
            }
        }
    }
    return []int{}
}

func main() {
    nums := []int{2, 7, 11, 15}
    target := 9
    
    result := twoSum(nums, target)
    fmt.Printf("Two Sum: %v\n", result) // [0, 1]
    
    result2 := twoSumBruteForce(nums, target)
    fmt.Printf("Two Sum (Brute Force): %v\n", result2) // [0, 1]
}
```

### **2. Maximum Subarray (Kadane's Algorithm)**
**Problem**: Find the contiguous subarray with maximum sum.

```go
package main

import "fmt"

// Time: O(n), Space: O(1)
func maxSubArray(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxSum := nums[0]
    currentSum := nums[0]
    
    for i := 1; i < len(nums); i++ {
        currentSum = max(nums[i], currentSum + nums[i])
        maxSum = max(maxSum, currentSum)
    }
    
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    nums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
    result := maxSubArray(nums)
    fmt.Printf("Maximum Subarray Sum: %d\n", result) // 6
}
```

### **3. Longest Substring Without Repeating Characters**
**Problem**: Find the length of the longest substring without repeating characters.

```go
package main

import "fmt"

// Time: O(n), Space: O(min(m,n)) where m is charset size
func lengthOfLongestSubstring(s string) int {
    charMap := make(map[byte]int)
    maxLen := 0
    left := 0
    
    for right := 0; right < len(s); right++ {
        if index, exists := charMap[s[right]]; exists && index >= left {
            left = index + 1
        }
        charMap[s[right]] = right
        maxLen = max(maxLen, right - left + 1)
    }
    
    return maxLen
}

func main() {
    s := "abcabcbb"
    result := lengthOfLongestSubstring(s)
    fmt.Printf("Longest Substring Length: %d\n", result) // 3
}
```

---

## 🔗 Linked List Problems

### **4. Reverse Linked List**
**Problem**: Reverse a singly linked list.

```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

// Time: O(n), Space: O(1)
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

// Time: O(n), Space: O(n) - Recursive approach
func reverseListRecursive(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    
    newHead := reverseListRecursive(head.Next)
    head.Next.Next = head
    head.Next = nil
    
    return newHead
}

func printList(head *ListNode) {
    for head != nil {
        fmt.Printf("%d -> ", head.Val)
        head = head.Next
    }
    fmt.Println("nil")
}

func main() {
    // Create list: 1 -> 2 -> 3 -> 4 -> 5
    head := &ListNode{Val: 1}
    head.Next = &ListNode{Val: 2}
    head.Next.Next = &ListNode{Val: 3}
    head.Next.Next.Next = &ListNode{Val: 4}
    head.Next.Next.Next.Next = &ListNode{Val: 5}
    
    fmt.Print("Original: ")
    printList(head)
    
    reversed := reverseList(head)
    fmt.Print("Reversed: ")
    printList(reversed)
}
```

### **5. Merge Two Sorted Lists**
**Problem**: Merge two sorted linked lists into one sorted list.

```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

// Time: O(n + m), Space: O(1)
func mergeTwoLists(l1, l2 *ListNode) *ListNode {
    dummy := &ListNode{}
    current := dummy
    
    for l1 != nil && l2 != nil {
        if l1.Val <= l2.Val {
            current.Next = l1
            l1 = l1.Next
        } else {
            current.Next = l2
            l2 = l2.Next
        }
        current = current.Next
    }
    
    // Attach remaining nodes
    if l1 != nil {
        current.Next = l1
    } else {
        current.Next = l2
    }
    
    return dummy.Next
}

func printList(head *ListNode) {
    for head != nil {
        fmt.Printf("%d -> ", head.Val)
        head = head.Next
    }
    fmt.Println("nil")
}

func main() {
    // List 1: 1 -> 2 -> 4
    l1 := &ListNode{Val: 1}
    l1.Next = &ListNode{Val: 2}
    l1.Next.Next = &ListNode{Val: 4}
    
    // List 2: 1 -> 3 -> 4
    l2 := &ListNode{Val: 1}
    l2.Next = &ListNode{Val: 3}
    l2.Next.Next = &ListNode{Val: 4}
    
    fmt.Print("List 1: ")
    printList(l1)
    fmt.Print("List 2: ")
    printList(l2)
    
    merged := mergeTwoLists(l1, l2)
    fmt.Print("Merged: ")
    printList(merged)
}
```

---

## 🌳 Tree & Graph Problems

### **6. Binary Tree Inorder Traversal**
**Problem**: Traverse a binary tree in inorder (left, root, right).

```go
package main

import "fmt"

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Time: O(n), Space: O(h) where h is height
func inorderTraversal(root *TreeNode) []int {
    var result []int
    var stack []*TreeNode
    current := root
    
    for current != nil || len(stack) > 0 {
        // Go to leftmost node
        for current != nil {
            stack = append(stack, current)
            current = current.Left
        }
        
        // Process current node
        current = stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        result = append(result, current.Val)
        
        // Move to right subtree
        current = current.Right
    }
    
    return result
}

// Time: O(n), Space: O(h) - Recursive approach
func inorderTraversalRecursive(root *TreeNode) []int {
    var result []int
    
    var helper func(*TreeNode)
    helper = func(node *TreeNode) {
        if node == nil {
            return
        }
        helper(node.Left)
        result = append(result, node.Val)
        helper(node.Right)
    }
    
    helper(root)
    return result
}

func main() {
    // Create tree:    1
    //                / \
    //               2   3
    //              / \
    //             4   5
    root := &TreeNode{Val: 1}
    root.Left = &TreeNode{Val: 2}
    root.Right = &TreeNode{Val: 3}
    root.Left.Left = &TreeNode{Val: 4}
    root.Left.Right = &TreeNode{Val: 5}
    
    result := inorderTraversal(root)
    fmt.Printf("Inorder Traversal: %v\n", result) // [4, 2, 5, 1, 3]
    
    result2 := inorderTraversalRecursive(root)
    fmt.Printf("Inorder Traversal (Recursive): %v\n", result2) // [4, 2, 5, 1, 3]
}
```

### **7. Maximum Depth of Binary Tree**
**Problem**: Find the maximum depth of a binary tree.

```go
package main

import "fmt"

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Time: O(n), Space: O(h) - Recursive approach
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    return max(leftDepth, rightDepth) + 1
}

// Time: O(n), Space: O(w) where w is max width - Iterative approach
func maxDepthIterative(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    queue := []*TreeNode{root}
    depth := 0
    
    for len(queue) > 0 {
        levelSize := len(queue)
        depth++
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
    }
    
    return depth
}

func main() {
    // Create tree:    3
    //                / \
    //               9   20
    //                  / \
    //                 15  7
    root := &TreeNode{Val: 3}
    root.Left = &TreeNode{Val: 9}
    root.Right = &TreeNode{Val: 20}
    root.Right.Left = &TreeNode{Val: 15}
    root.Right.Right = &TreeNode{Val: 7}
    
    depth := maxDepth(root)
    fmt.Printf("Maximum Depth: %d\n", depth) // 3
    
    depth2 := maxDepthIterative(root)
    fmt.Printf("Maximum Depth (Iterative): %d\n", depth2) // 3
}
```

---

## 💡 Dynamic Programming

### **8. Fibonacci Sequence**
**Problem**: Calculate the nth Fibonacci number.

```go
package main

import "fmt"

// Time: O(n), Space: O(1) - Optimized
func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    
    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a+b
    }
    
    return b
}

// Time: O(n), Space: O(n) - DP with memoization
func fibonacciDP(n int) int {
    if n <= 1 {
        return n
    }
    
    dp := make([]int, n+1)
    dp[0], dp[1] = 0, 1
    
    for i := 2; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    
    return dp[n]
}

// Time: O(2^n), Space: O(n) - Naive recursive
func fibonacciRecursive(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacciRecursive(n-1) + fibonacciRecursive(n-2)
}

func main() {
    n := 10
    
    result := fibonacci(n)
    fmt.Printf("Fibonacci(%d) = %d\n", n, result) // 55
    
    result2 := fibonacciDP(n)
    fmt.Printf("Fibonacci DP(%d) = %d\n", n, result2) // 55
    
    result3 := fibonacciRecursive(n)
    fmt.Printf("Fibonacci Recursive(%d) = %d\n", n, result3) // 55
}
```

### **9. Longest Common Subsequence**
**Problem**: Find the length of the longest common subsequence between two strings.

```go
package main

import "fmt"

// Time: O(m*n), Space: O(m*n)
func longestCommonSubsequence(text1, text2 string) int {
    m, n := len(text1), len(text2)
    dp := make([][]int, m+1)
    for i := range dp {
        dp[i] = make([]int, n+1)
    }
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                dp[i][j] = dp[i-1][j-1] + 1
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            }
        }
    }
    
    return dp[m][n]
}

// Time: O(m*n), Space: O(min(m,n)) - Space optimized
func longestCommonSubsequenceOptimized(text1, text2 string) int {
    if len(text1) < len(text2) {
        text1, text2 = text2, text1
    }
    
    m, n := len(text1), len(text2)
    prev := make([]int, n+1)
    curr := make([]int, n+1)
    
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            if text1[i-1] == text2[j-1] {
                curr[j] = prev[j-1] + 1
            } else {
                curr[j] = max(prev[j], curr[j-1])
            }
        }
        prev, curr = curr, prev
    }
    
    return prev[n]
}

func main() {
    text1 := "abcde"
    text2 := "ace"
    
    result := longestCommonSubsequence(text1, text2)
    fmt.Printf("LCS Length: %d\n", result) // 3
    
    result2 := longestCommonSubsequenceOptimized(text1, text2)
    fmt.Printf("LCS Length (Optimized): %d\n", result2) // 3
}
```

---

## 🔍 Sorting & Searching

### **10. Quick Sort**
**Problem**: Implement quick sort algorithm.

```go
package main

import "fmt"

// Time: O(n log n) average, O(n²) worst case, Space: O(log n)
func quickSort(arr []int) {
    if len(arr) <= 1 {
        return
    }
    
    pivot := partition(arr)
    quickSort(arr[:pivot])
    quickSort(arr[pivot+1:])
}

func partition(arr []int) int {
    pivot := arr[len(arr)-1]
    i := 0
    
    for j := 0; j < len(arr)-1; j++ {
        if arr[j] < pivot {
            arr[i], arr[j] = arr[j], arr[i]
            i++
        }
    }
    
    arr[i], arr[len(arr)-1] = arr[len(arr)-1], arr[i]
    return i
}

func main() {
    arr := []int{64, 34, 25, 12, 22, 11, 90}
    fmt.Printf("Original: %v\n", arr)
    
    quickSort(arr)
    fmt.Printf("Sorted: %v\n", arr)
}
```

### **11. Binary Search**
**Problem**: Search for a target value in a sorted array.

```go
package main

import "fmt"

// Time: O(log n), Space: O(1)
func binarySearch(nums []int, target int) int {
    left, right := 0, len(nums)-1
    
    for left <= right {
        mid := left + (right-left)/2
        
        if nums[mid] == target {
            return mid
        } else if nums[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

// Time: O(log n), Space: O(log n) - Recursive approach
func binarySearchRecursive(nums []int, target int) int {
    return binarySearchHelper(nums, target, 0, len(nums)-1)
}

func binarySearchHelper(nums []int, target, left, right int) int {
    if left > right {
        return -1
    }
    
    mid := left + (right-left)/2
    
    if nums[mid] == target {
        return mid
    } else if nums[mid] < target {
        return binarySearchHelper(nums, target, mid+1, right)
    } else {
        return binarySearchHelper(nums, target, left, mid-1)
    }
}

func main() {
    nums := []int{1, 3, 5, 7, 9, 11, 13, 15}
    target := 7
    
    result := binarySearch(nums, target)
    fmt.Printf("Binary Search: %d\n", result) // 3
    
    result2 := binarySearchRecursive(nums, target)
    fmt.Printf("Binary Search (Recursive): %d\n", result2) // 3
}
```

---

## 🎯 FAANG Interview Questions

### **Google Interview Questions**

#### **1. Spiral Matrix**
**Question**: "Given a matrix, return all elements in spiral order."

**Answer**:
```go
package main

import "fmt"

// Time: O(m*n), Space: O(1)
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 || len(matrix[0]) == 0 {
        return []int{}
    }
    
    rows, cols := len(matrix), len(matrix[0])
    result := make([]int, 0, rows*cols)
    
    top, bottom := 0, rows-1
    left, right := 0, cols-1
    
    for top <= bottom && left <= right {
        // Traverse right
        for col := left; col <= right; col++ {
            result = append(result, matrix[top][col])
        }
        top++
        
        // Traverse down
        for row := top; row <= bottom; row++ {
            result = append(result, matrix[row][right])
        }
        right--
        
        // Traverse left
        if top <= bottom {
            for col := right; col >= left; col-- {
                result = append(result, matrix[bottom][col])
            }
            bottom--
        }
        
        // Traverse up
        if left <= right {
            for row := bottom; row >= top; row-- {
                result = append(result, matrix[row][left])
            }
            left++
        }
    }
    
    return result
}

func main() {
    matrix := [][]int{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    }
    
    result := spiralOrder(matrix)
    fmt.Printf("Spiral Order: %v\n", result) // [1, 2, 3, 6, 9, 8, 7, 4, 5]
}
```

### **Meta Interview Questions**

#### **2. Valid Parentheses**
**Question**: "Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid."

**Answer**:
```go
package main

import "fmt"

// Time: O(n), Space: O(n)
func isValid(s string) bool {
    stack := make([]rune, 0)
    pairs := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            if len(stack) == 0 || stack[len(stack)-1] != pairs[char] {
                return false
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}

func main() {
    testCases := []string{
        "()",
        "()[]{}",
        "(]",
        "([)]",
        "{[]}",
    }
    
    for _, test := range testCases {
        result := isValid(test)
        fmt.Printf("'%s' is valid: %t\n", test, result)
    }
}
```

### **Amazon Interview Questions**

#### **3. Product of Array Except Self**
**Question**: "Given an array nums, return an array where each element is the product of all elements except itself."

**Answer**:
```go
package main

import "fmt"

// Time: O(n), Space: O(1) - excluding output array
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // First pass: calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Second pass: calculate right products and multiply
    rightProduct := 1
    for i := n - 1; i >= 0; i-- {
        result[i] *= rightProduct
        rightProduct *= nums[i]
    }
    
    return result
}

func main() {
    nums := []int{1, 2, 3, 4}
    result := productExceptSelf(nums)
    fmt.Printf("Product Except Self: %v\n", result) // [24, 12, 8, 6]
}
```

---

## 📚 Additional Resources

### **Books**
- [Cracking the Coding Interview](https://www.crackingthecodinginterview.com/) - Gayle Laakmann McDowell
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms) - Thomas H. Cormen
- [Algorithm Design Manual](https://www.algorist.com/) - Steven S. Skiena

### **Online Resources**
- [LeetCode](https://leetcode.com/) - Practice problems
- [HackerRank](https://www.hackerrank.com/) - Coding challenges
- [GeeksforGeeks](https://www.geeksforgeeks.org/) - Algorithm explanations

### **Video Resources**
- [Abdul Bari](https://www.youtube.com/c/AbdulBari) - Algorithm explanations
- [Back To Back SWE](https://www.youtube.com/c/BackToBackSWE) - Interview preparation
- [Tech Interview Pro](https://www.youtube.com/c/TechInterviewPro) - Coding interviews

---

*This comprehensive guide covers essential DSA problems with optimal Go implementations and real-world interview questions from top tech companies.*
