---
# Auto-generated front matter
Title: Dsa-Questions-Golang-Nodejs
LastUpdated: 2025-11-06T20:45:58.419325
Tags: []
Status: draft
---

# Data Structures & Algorithms - Golang & Node.js

## Table of Contents

1. [Overview](#overview)
2. [Arrays and Strings](#arrays-and-strings)
3. [Linked Lists](#linked-lists)
4. [Stacks and Queues](#stacks-and-queues)
5. [Trees](#trees)
6. [Graphs](#graphs)
7. [Dynamic Programming](#dynamic-programming)
8. [Sorting and Searching](#sorting-and-searching)
9. [Hash Tables](#hash-tables)
10. [Complexity Analysis](#complexity-analysis)

## Overview

### Learning Objectives

- Master fundamental data structures
- Solve algorithmic problems efficiently
- Understand time and space complexity
- Implement solutions in both Go and Node.js
- Apply problem-solving patterns

### Problem Categories

- **Arrays & Strings**: Two pointers, sliding window, hash maps
- **Linked Lists**: Traversal, manipulation, cycle detection
- **Stacks & Queues**: LIFO/FIFO operations, monotonic stacks
- **Trees**: Traversal, BST operations, tree construction
- **Graphs**: DFS/BFS, shortest paths, topological sort
- **DP**: Memoization, tabulation, optimization
- **Sorting**: Comparison-based, non-comparison sorting
- **Searching**: Binary search, interpolation search

## Arrays and Strings

### 1. Two Sum

#### Problem
Given an array of integers and a target sum, return indices of two numbers that add up to the target.

#### Go Solution
```go
package main

import "fmt"

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

func main() {
    nums := []int{2, 7, 11, 15}
    target := 9
    result := twoSum(nums, target)
    fmt.Println(result) // [0, 1]
}
```

#### Node.js Solution
```javascript
function twoSum(nums, target) {
    const numMap = new Map();
    
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (numMap.has(complement)) {
            return [numMap.get(complement), i];
        }
        numMap.set(nums[i], i);
    }
    
    return [];
}

// Example
const nums = [2, 7, 11, 15];
const target = 9;
console.log(twoSum(nums, target)); // [0, 1]
```

### 2. Maximum Subarray (Kadane's Algorithm)

#### Problem
Find the contiguous subarray with maximum sum.

#### Go Solution
```go
package main

import (
    "fmt"
    "math"
)

func maxSubArray(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    maxSoFar := nums[0]
    maxEndingHere := nums[0]
    
    for i := 1; i < len(nums); i++ {
        maxEndingHere = int(math.Max(float64(nums[i]), float64(maxEndingHere + nums[i])))
        maxSoFar = int(math.Max(float64(maxSoFar), float64(maxEndingHere)))
    }
    
    return maxSoFar
}

func main() {
    nums := []int{-2, 1, -3, 4, -1, 2, 1, -5, 4}
    result := maxSubArray(nums)
    fmt.Println(result) // 6
}
```

#### Node.js Solution
```javascript
function maxSubArray(nums) {
    if (nums.length === 0) return 0;
    
    let maxSoFar = nums[0];
    let maxEndingHere = nums[0];
    
    for (let i = 1; i < nums.length; i++) {
        maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    
    return maxSoFar;
}

// Example
const nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
console.log(maxSubArray(nums)); // 6
```

### 3. Valid Parentheses

#### Problem
Check if a string of parentheses is valid.

#### Go Solution
```go
package main

import "fmt"

func isValid(s string) bool {
    stack := []rune{}
    mapping := map[rune]rune{
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for _, char := range s {
        if char == '(' || char == '{' || char == '[' {
            stack = append(stack, char)
        } else if char == ')' || char == '}' || char == ']' {
            if len(stack) == 0 {
                return false
            }
            if stack[len(stack)-1] != mapping[char] {
                return false
            }
            stack = stack[:len(stack)-1]
        }
    }
    
    return len(stack) == 0
}

func main() {
    testCases := []string{"()", "()[]{}", "(]", "([)]", "{[]}"}
    for _, test := range testCases {
        fmt.Printf("%s: %t\n", test, isValid(test))
    }
}
```

#### Node.js Solution
```javascript
function isValid(s) {
    const stack = [];
    const mapping = {
        ')': '(',
        '}': '{',
        ']': '['
    };
    
    for (const char of s) {
        if (char === '(' || char === '{' || char === '[') {
            stack.push(char);
        } else if (char === ')' || char === '}' || char === ']') {
            if (stack.length === 0) return false;
            if (stack.pop() !== mapping[char]) return false;
        }
    }
    
    return stack.length === 0;
}

// Example
const testCases = ["()", "()[]{}", "(]", "([)]", "{[]}"];
testCases.forEach(test => {
    console.log(`${test}: ${isValid(test)}`);
});
```

## Linked Lists

### 1. Reverse Linked List

#### Problem
Reverse a singly linked list.

#### Go Solution
```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

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

func printList(head *ListNode) {
    current := head
    for current != nil {
        fmt.Printf("%d -> ", current.Val)
        current = current.Next
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

#### Node.js Solution
```javascript
class ListNode {
    constructor(val, next = null) {
        this.val = val;
        this.next = next;
    }
}

function reverseList(head) {
    let prev = null;
    let current = head;
    
    while (current !== null) {
        const next = current.next;
        current.next = prev;
        prev = current;
        current = next;
    }
    
    return prev;
}

function printList(head) {
    let current = head;
    let result = '';
    while (current !== null) {
        result += current.val + ' -> ';
        current = current.next;
    }
    console.log(result + 'null');
}

// Example
const head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
head.next.next.next = new ListNode(4);
head.next.next.next.next = new ListNode(5);

console.log('Original:');
printList(head);

const reversed = reverseList(head);
console.log('Reversed:');
printList(reversed);
```

### 2. Detect Cycle in Linked List

#### Problem
Detect if a linked list has a cycle using Floyd's algorithm.

#### Go Solution
```go
package main

import "fmt"

type ListNode struct {
    Val  int
    Next *ListNode
}

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

func main() {
    // Create list with cycle: 1 -> 2 -> 3 -> 4 -> 2
    head := &ListNode{Val: 1}
    node2 := &ListNode{Val: 2}
    node3 := &ListNode{Val: 3}
    node4 := &ListNode{Val: 4}
    
    head.Next = node2
    node2.Next = node3
    node3.Next = node4
    node4.Next = node2 // Cycle
    
    fmt.Println("Has cycle:", hasCycle(head)) // true
}
```

#### Node.js Solution
```javascript
class ListNode {
    constructor(val, next = null) {
        this.val = val;
        this.next = next;
    }
}

function hasCycle(head) {
    if (!head || !head.next) return false;
    
    let slow = head;
    let fast = head.next;
    
    while (fast && fast.next) {
        if (slow === fast) return true;
        slow = slow.next;
        fast = fast.next.next;
    }
    
    return false;
}

// Example
const head = new ListNode(1);
const node2 = new ListNode(2);
const node3 = new ListNode(3);
const node4 = new ListNode(4);

head.next = node2;
node2.next = node3;
node3.next = node4;
node4.next = node2; // Cycle

console.log('Has cycle:', hasCycle(head)); // true
```

## Stacks and Queues

### 1. Implement Stack using Array

#### Problem
Implement a stack with push, pop, top, and isEmpty operations.

#### Go Solution
```go
package main

import "fmt"

type Stack struct {
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() int {
    if s.IsEmpty() {
        return -1 // or panic
    }
    
    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]
    return item
}

func (s *Stack) Top() int {
    if s.IsEmpty() {
        return -1 // or panic
    }
    
    return s.items[len(s.items)-1]
}

func (s *Stack) IsEmpty() bool {
    return len(s.items) == 0
}

func (s *Stack) Size() int {
    return len(s.items)
}

func main() {
    stack := &Stack{}
    
    stack.Push(1)
    stack.Push(2)
    stack.Push(3)
    
    fmt.Println("Top:", stack.Top()) // 3
    fmt.Println("Size:", stack.Size()) // 3
    
    fmt.Println("Pop:", stack.Pop()) // 3
    fmt.Println("Pop:", stack.Pop()) // 2
    fmt.Println("IsEmpty:", stack.IsEmpty()) // false
    
    fmt.Println("Pop:", stack.Pop()) // 1
    fmt.Println("IsEmpty:", stack.IsEmpty()) // true
}
```

#### Node.js Solution
```javascript
class Stack {
    constructor() {
        this.items = [];
    }
    
    push(item) {
        this.items.push(item);
    }
    
    pop() {
        if (this.isEmpty()) {
            return -1; // or throw error
        }
        return this.items.pop();
    }
    
    top() {
        if (this.isEmpty()) {
            return -1; // or throw error
        }
        return this.items[this.items.length - 1];
    }
    
    isEmpty() {
        return this.items.length === 0;
    }
    
    size() {
        return this.items.length;
    }
}

// Example
const stack = new Stack();

stack.push(1);
stack.push(2);
stack.push(3);

console.log('Top:', stack.top()); // 3
console.log('Size:', stack.size()); // 3

console.log('Pop:', stack.pop()); // 3
console.log('Pop:', stack.pop()); // 2
console.log('IsEmpty:', stack.isEmpty()); // false

console.log('Pop:', stack.pop()); // 1
console.log('IsEmpty:', stack.isEmpty()); // true
```

### 2. Implement Queue using Array

#### Problem
Implement a queue with enqueue, dequeue, front, and isEmpty operations.

#### Go Solution
```go
package main

import "fmt"

type Queue struct {
    items []int
}

func (q *Queue) Enqueue(item int) {
    q.items = append(q.items, item)
}

func (q *Queue) Dequeue() int {
    if q.IsEmpty() {
        return -1 // or panic
    }
    
    item := q.items[0]
    q.items = q.items[1:]
    return item
}

func (q *Queue) Front() int {
    if q.IsEmpty() {
        return -1 // or panic
    }
    
    return q.items[0]
}

func (q *Queue) IsEmpty() bool {
    return len(q.items) == 0
}

func (q *Queue) Size() int {
    return len(q.items)
}

func main() {
    queue := &Queue{}
    
    queue.Enqueue(1)
    queue.Enqueue(2)
    queue.Enqueue(3)
    
    fmt.Println("Front:", queue.Front()) // 1
    fmt.Println("Size:", queue.Size()) // 3
    
    fmt.Println("Dequeue:", queue.Dequeue()) // 1
    fmt.Println("Dequeue:", queue.Dequeue()) // 2
    fmt.Println("IsEmpty:", queue.IsEmpty()) // false
    
    fmt.Println("Dequeue:", queue.Dequeue()) // 3
    fmt.Println("IsEmpty:", queue.IsEmpty()) // true
}
```

#### Node.js Solution
```javascript
class Queue {
    constructor() {
        this.items = [];
    }
    
    enqueue(item) {
        this.items.push(item);
    }
    
    dequeue() {
        if (this.isEmpty()) {
            return -1; // or throw error
        }
        return this.items.shift();
    }
    
    front() {
        if (this.isEmpty()) {
            return -1; // or throw error
        }
        return this.items[0];
    }
    
    isEmpty() {
        return this.items.length === 0;
    }
    
    size() {
        return this.items.length;
    }
}

// Example
const queue = new Queue();

queue.enqueue(1);
queue.enqueue(2);
queue.enqueue(3);

console.log('Front:', queue.front()); // 1
console.log('Size:', queue.size()); // 3

console.log('Dequeue:', queue.dequeue()); // 1
console.log('Dequeue:', queue.dequeue()); // 2
console.log('IsEmpty:', queue.isEmpty()); // false

console.log('Dequeue:', queue.dequeue()); // 3
console.log('IsEmpty:', queue.isEmpty()); // true
```

## Trees

### 1. Binary Tree Traversal

#### Problem
Implement preorder, inorder, and postorder traversal of a binary tree.

#### Go Solution
```go
package main

import "fmt"

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Preorder: Root -> Left -> Right
func preorderTraversal(root *TreeNode) []int {
    var result []int
    
    var preorder func(*TreeNode)
    preorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        result = append(result, node.Val)
        preorder(node.Left)
        preorder(node.Right)
    }
    
    preorder(root)
    return result
}

// Inorder: Left -> Root -> Right
func inorderTraversal(root *TreeNode) []int {
    var result []int
    
    var inorder func(*TreeNode)
    inorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        inorder(node.Left)
        result = append(result, node.Val)
        inorder(node.Right)
    }
    
    inorder(root)
    return result
}

// Postorder: Left -> Right -> Root
func postorderTraversal(root *TreeNode) []int {
    var result []int
    
    var postorder func(*TreeNode)
    postorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        postorder(node.Left)
        postorder(node.Right)
        result = append(result, node.Val)
    }
    
    postorder(root)
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
    
    fmt.Println("Preorder:", preorderTraversal(root))  // [1, 2, 4, 5, 3]
    fmt.Println("Inorder:", inorderTraversal(root))    // [4, 2, 5, 1, 3]
    fmt.Println("Postorder:", postorderTraversal(root)) // [4, 5, 2, 3, 1]
}
```

#### Node.js Solution
```javascript
class TreeNode {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// Preorder: Root -> Left -> Right
function preorderTraversal(root) {
    const result = [];
    
    function preorder(node) {
        if (!node) return;
        result.push(node.val);
        preorder(node.left);
        preorder(node.right);
    }
    
    preorder(root);
    return result;
}

// Inorder: Left -> Root -> Right
function inorderTraversal(root) {
    const result = [];
    
    function inorder(node) {
        if (!node) return;
        inorder(node.left);
        result.push(node.val);
        inorder(node.right);
    }
    
    inorder(root);
    return result;
}

// Postorder: Left -> Right -> Root
function postorderTraversal(root) {
    const result = [];
    
    function postorder(node) {
        if (!node) return;
        postorder(node.left);
        postorder(node.right);
        result.push(node.val);
    }
    
    postorder(root);
    return result;
}

// Example
const root = new TreeNode(1);
root.left = new TreeNode(2);
root.right = new TreeNode(3);
root.left.left = new TreeNode(4);
root.left.right = new TreeNode(5);

console.log('Preorder:', preorderTraversal(root));  // [1, 2, 4, 5, 3]
console.log('Inorder:', inorderTraversal(root));    // [4, 2, 5, 1, 3]
console.log('Postorder:', postorderTraversal(root)); // [4, 5, 2, 3, 1]
```

### 2. Maximum Depth of Binary Tree

#### Problem
Find the maximum depth of a binary tree.

#### Go Solution
```go
package main

import "fmt"

type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    if leftDepth > rightDepth {
        return leftDepth + 1
    }
    return rightDepth + 1
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
    
    fmt.Println("Max depth:", maxDepth(root)) // 3
}
```

#### Node.js Solution
```javascript
class TreeNode {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function maxDepth(root) {
    if (!root) return 0;
    
    const leftDepth = maxDepth(root.left);
    const rightDepth = maxDepth(root.right);
    
    return Math.max(leftDepth, rightDepth) + 1;
}

// Example
const root = new TreeNode(1);
root.left = new TreeNode(2);
root.right = new TreeNode(3);
root.left.left = new TreeNode(4);
root.left.right = new TreeNode(5);

console.log('Max depth:', maxDepth(root)); // 3
```

## Dynamic Programming

### 1. Fibonacci Numbers

#### Problem
Calculate the nth Fibonacci number using dynamic programming.

#### Go Solution
```go
package main

import "fmt"

// Memoization approach
func fibonacciMemo(n int) int {
    memo := make(map[int]int)
    return fibMemo(n, memo)
}

func fibMemo(n int, memo map[int]int) int {
    if n <= 1 {
        return n
    }
    
    if val, exists := memo[n]; exists {
        return val
    }
    
    memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo)
    return memo[n]
}

// Tabulation approach
func fibonacciTab(n int) int {
    if n <= 1 {
        return n
    }
    
    dp := make([]int, n+1)
    dp[0] = 0
    dp[1] = 1
    
    for i := 2; i <= n; i++ {
        dp[i] = dp[i-1] + dp[i-2]
    }
    
    return dp[n]
}

// Space-optimized approach
func fibonacciOpt(n int) int {
    if n <= 1 {
        return n
    }
    
    prev2 := 0
    prev1 := 1
    
    for i := 2; i <= n; i++ {
        current := prev1 + prev2
        prev2 = prev1
        prev1 = current
    }
    
    return prev1
}

func main() {
    n := 10
    fmt.Printf("Fibonacci(%d) = %d (memo)\n", n, fibonacciMemo(n))
    fmt.Printf("Fibonacci(%d) = %d (tab)\n", n, fibonacciTab(n))
    fmt.Printf("Fibonacci(%d) = %d (opt)\n", n, fibonacciOpt(n))
}
```

#### Node.js Solution
```javascript
// Memoization approach
function fibonacciMemo(n, memo = {}) {
    if (n <= 1) return n;
    if (memo[n]) return memo[n];
    
    memo[n] = fibonacciMemo(n - 1, memo) + fibonacciMemo(n - 2, memo);
    return memo[n];
}

// Tabulation approach
function fibonacciTab(n) {
    if (n <= 1) return n;
    
    const dp = [0, 1];
    
    for (let i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}

// Space-optimized approach
function fibonacciOpt(n) {
    if (n <= 1) return n;
    
    let prev2 = 0;
    let prev1 = 1;
    
    for (let i = 2; i <= n; i++) {
        const current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}

// Example
const n = 10;
console.log(`Fibonacci(${n}) = ${fibonacciMemo(n)} (memo)`);
console.log(`Fibonacci(${n}) = ${fibonacciTab(n)} (tab)`);
console.log(`Fibonacci(${n}) = ${fibonacciOpt(n)} (opt)`);
```

### 2. Longest Common Subsequence

#### Problem
Find the length of the longest common subsequence between two strings.

#### Go Solution
```go
package main

import "fmt"

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

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func main() {
    text1 := "abcde"
    text2 := "ace"
    result := longestCommonSubsequence(text1, text2)
    fmt.Printf("LCS length: %d\n", result) // 3
}
```

#### Node.js Solution
```javascript
function longestCommonSubsequence(text1, text2) {
    const m = text1.length;
    const n = text2.length;
    const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0));
    
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (text1[i - 1] === text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    return dp[m][n];
}

// Example
const text1 = "abcde";
const text2 = "ace";
const result = longestCommonSubsequence(text1, text2);
console.log(`LCS length: ${result}`); // 3
```

## Complexity Analysis

### 1. Time Complexity

```go
// O(1) - Constant time
func getFirstElement(arr []int) int {
    return arr[0]
}

// O(n) - Linear time
func linearSearch(arr []int, target int) int {
    for i, val := range arr {
        if val == target {
            return i
        }
    }
    return -1
}

// O(n²) - Quadratic time
func bubbleSort(arr []int) {
    n := len(arr)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
}

// O(log n) - Logarithmic time
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}
```

### 2. Space Complexity

```go
// O(1) - Constant space
func swap(a, b *int) {
    *a, *b = *b, *a
}

// O(n) - Linear space
func createArray(n int) []int {
    arr := make([]int, n)
    for i := 0; i < n; i++ {
        arr[i] = i
    }
    return arr
}

// O(n²) - Quadratic space
func createMatrix(n int) [][]int {
    matrix := make([][]int, n)
    for i := 0; i < n; i++ {
        matrix[i] = make([]int, n)
    }
    return matrix
}
```

## Follow-up Questions

### 1. Algorithm Design
**Q: How do you approach a new algorithmic problem?**
A: 1) Understand the problem and constraints, 2) Identify patterns and data structures, 3) Start with brute force, 4) Optimize using known techniques, 5) Consider edge cases and test thoroughly.

### 2. Data Structure Selection
**Q: When would you use a hash table vs a binary search tree?**
A: Use hash tables for O(1) average-case operations and when order doesn't matter. Use BSTs when you need ordered data, range queries, or guaranteed O(log n) performance.

### 3. Complexity Analysis
**Q: What's the difference between average-case and worst-case complexity?**
A: Average-case is the expected performance over all possible inputs, while worst-case is the maximum time/space required for any input. Worst-case is more important for critical systems.

## Sources

### Books
- **Introduction to Algorithms** by CLRS
- **Cracking the Coding Interview** by Gayle Laakmann McDowell
- **Elements of Programming Interviews** by Aziz, Lee, and Prakash

### Online Resources
- **LeetCode** - Algorithm practice platform
- **HackerRank** - Coding challenges
- **GeeksforGeeks** - Algorithm explanations

## Projects

### 1. Algorithm Visualizer
**Objective**: Build a tool to visualize sorting algorithms
**Requirements**: Multiple sorting algorithms, step-by-step visualization
**Deliverables**: Interactive algorithm visualizer

### 2. Data Structure Library
**Objective**: Implement common data structures from scratch
**Requirements**: Arrays, linked lists, trees, graphs, hash tables
**Deliverables**: Complete data structure library

### 3. Problem Solver
**Objective**: Create a tool to solve common algorithmic problems
**Requirements**: Multiple problem categories, solution explanations
**Deliverables**: Comprehensive problem-solving tool

---

**Next**: [Design Patterns](design-patterns.md) | **Previous**: [Node.js Fundamentals](nodejs-fundamentals.md) | **Up**: [Phase 0](README.md)



## Graphs

<!-- AUTO-GENERATED ANCHOR: originally referenced as #graphs -->

Placeholder content. Please replace with proper section.


## Sorting And Searching

<!-- AUTO-GENERATED ANCHOR: originally referenced as #sorting-and-searching -->

Placeholder content. Please replace with proper section.


## Hash Tables

<!-- AUTO-GENERATED ANCHOR: originally referenced as #hash-tables -->

Placeholder content. Please replace with proper section.
