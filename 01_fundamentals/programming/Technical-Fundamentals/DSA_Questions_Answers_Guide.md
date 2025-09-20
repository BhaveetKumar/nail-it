# 🧮 DSA Questions and Answers Guide - Node.js

> **Comprehensive Data Structures and Algorithms guide with Node.js implementations**

## 🎯 **Overview**

This guide covers essential data structures and algorithms with detailed explanations, implementations, and complexity analysis. Each concept includes multiple approaches, real-world applications, and interview-style questions.

## 📚 **Table of Contents**

1. [Arrays and Strings](#arrays-and-strings)
2. [Linked Lists](#linked-lists)
3. [Stacks and Queues](#stacks-and-queues)
4. [Trees and Graphs](#trees-and-graphs)
5. [Dynamic Programming](#dynamic-programming)
6. [Sorting and Searching](#sorting-and-searching)
7. [Hash Tables and Sets](#hash-tables-and-sets)
8. [Advanced Algorithms](#advanced-algorithms)

---

## 📊 **Arrays and Strings**

### **Array Fundamentals**

```javascript
// Array Operations and Patterns
class ArrayOperations {
    // Two Pointers Technique
    static twoSum(nums, target) {
        const map = new Map();
        
        for (let i = 0; i < nums.length; i++) {
            const complement = target - nums[i];
            
            if (map.has(complement)) {
                return [map.get(complement), i];
            }
            
            map.set(nums[i], i);
        }
        
        return [];
    }
    
    // Sliding Window Technique
    static maxSubarraySum(arr, k) {
        if (arr.length < k) return null;
        
        let maxSum = 0;
        let windowSum = 0;
        
        // Calculate sum of first window
        for (let i = 0; i < k; i++) {
            windowSum += arr[i];
        }
        
        maxSum = windowSum;
        
        // Slide the window
        for (let i = k; i < arr.length; i++) {
            windowSum = windowSum - arr[i - k] + arr[i];
            maxSum = Math.max(maxSum, windowSum);
        }
        
        return maxSum;
    }
    
    // Dutch National Flag Problem
    static sortColors(nums) {
        let low = 0;
        let mid = 0;
        let high = nums.length - 1;
        
        while (mid <= high) {
            if (nums[mid] === 0) {
                [nums[low], nums[mid]] = [nums[mid], nums[low]];
                low++;
                mid++;
            } else if (nums[mid] === 1) {
                mid++;
            } else {
                [nums[mid], nums[high]] = [nums[high], nums[mid]];
                high--;
            }
        }
        
        return nums;
    }
    
    // Kadane's Algorithm for Maximum Subarray
    static maxSubarray(nums) {
        let maxSoFar = nums[0];
        let maxEndingHere = nums[0];
        
        for (let i = 1; i < nums.length; i++) {
            maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
            maxSoFar = Math.max(maxSoFar, maxEndingHere);
        }
        
        return maxSoFar;
    }
    
    // Product of Array Except Self
    static productExceptSelf(nums) {
        const result = new Array(nums.length);
        
        // Calculate left products
        result[0] = 1;
        for (let i = 1; i < nums.length; i++) {
            result[i] = result[i - 1] * nums[i - 1];
        }
        
        // Calculate right products and multiply
        let rightProduct = 1;
        for (let i = nums.length - 1; i >= 0; i--) {
            result[i] = result[i] * rightProduct;
            rightProduct *= nums[i];
        }
        
        return result;
    }
}

// String Operations
class StringOperations {
    // Longest Substring Without Repeating Characters
    static lengthOfLongestSubstring(s) {
        const charMap = new Map();
        let maxLength = 0;
        let start = 0;
        
        for (let end = 0; end < s.length; end++) {
            if (charMap.has(s[end]) && charMap.get(s[end]) >= start) {
                start = charMap.get(s[end]) + 1;
            }
            
            charMap.set(s[end], end);
            maxLength = Math.max(maxLength, end - start + 1);
        }
        
        return maxLength;
    }
    
    // Valid Anagram
    static isAnagram(s, t) {
        if (s.length !== t.length) return false;
        
        const charCount = new Map();
        
        // Count characters in first string
        for (const char of s) {
            charCount.set(char, (charCount.get(char) || 0) + 1);
        }
        
        // Decrease count for characters in second string
        for (const char of t) {
            const count = charCount.get(char);
            if (!count) return false;
            charCount.set(char, count - 1);
        }
        
        return true;
    }
    
    // Longest Palindromic Substring
    static longestPalindrome(s) {
        if (!s || s.length < 1) return '';
        
        let start = 0;
        let end = 0;
        
        for (let i = 0; i < s.length; i++) {
            const len1 = this.expandAroundCenter(s, i, i);
            const len2 = this.expandAroundCenter(s, i, i + 1);
            const len = Math.max(len1, len2);
            
            if (len > end - start) {
                start = i - Math.floor((len - 1) / 2);
                end = i + Math.floor(len / 2);
            }
        }
        
        return s.substring(start, end + 1);
    }
    
    static expandAroundCenter(s, left, right) {
        while (left >= 0 && right < s.length && s[left] === s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
    
    // String Permutations
    static permute(s) {
        const result = [];
        const used = new Array(s.length).fill(false);
        const current = [];
        
        this.backtrack(s, current, used, result);
        return result;
    }
    
    static backtrack(s, current, used, result) {
        if (current.length === s.length) {
            result.push(current.join(''));
            return;
        }
        
        for (let i = 0; i < s.length; i++) {
            if (used[i]) continue;
            
            current.push(s[i]);
            used[i] = true;
            this.backtrack(s, current, used, result);
            current.pop();
            used[i] = false;
        }
    }
}
```

---

## 🔗 **Linked Lists**

### **Linked List Implementation**

```javascript
// Linked List Node
class ListNode {
    constructor(val, next = null) {
        this.val = val;
        this.next = next;
    }
}

// Linked List Operations
class LinkedListOperations {
    // Reverse Linked List
    static reverseList(head) {
        let prev = null;
        let current = head;
        
        while (current) {
            const next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
    
    // Merge Two Sorted Lists
    static mergeTwoLists(l1, l2) {
        const dummy = new ListNode(0);
        let current = dummy;
        
        while (l1 && l2) {
            if (l1.val <= l2.val) {
                current.next = l1;
                l1 = l1.next;
            } else {
                current.next = l2;
                l2 = l2.next;
            }
            current = current.next;
        }
        
        current.next = l1 || l2;
        return dummy.next;
    }
    
    // Detect Cycle in Linked List
    static hasCycle(head) {
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
    
    // Find Middle of Linked List
    static findMiddle(head) {
        let slow = head;
        let fast = head;
        
        while (fast && fast.next) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return slow;
    }
    
    // Remove Nth Node From End
    static removeNthFromEnd(head, n) {
        const dummy = new ListNode(0);
        dummy.next = head;
        
        let first = dummy;
        let second = dummy;
        
        // Move first n+1 steps ahead
        for (let i = 0; i <= n; i++) {
            first = first.next;
        }
        
        // Move both pointers until first reaches end
        while (first) {
            first = first.next;
            second = second.next;
        }
        
        second.next = second.next.next;
        return dummy.next;
    }
    
    // Add Two Numbers
    static addTwoNumbers(l1, l2) {
        const dummy = new ListNode(0);
        let current = dummy;
        let carry = 0;
        
        while (l1 || l2 || carry) {
            const sum = (l1?.val || 0) + (l2?.val || 0) + carry;
            carry = Math.floor(sum / 10);
            current.next = new ListNode(sum % 10);
            current = current.next;
            
            l1 = l1?.next;
            l2 = l2?.next;
        }
        
        return dummy.next;
    }
}

// Doubly Linked List
class DoublyListNode {
    constructor(val, prev = null, next = null) {
        this.val = val;
        this.prev = prev;
        this.next = next;
    }
}

class DoublyLinkedList {
    constructor() {
        this.head = new DoublyListNode(0);
        this.tail = new DoublyListNode(0);
        this.head.next = this.tail;
        this.tail.prev = this.head;
        this.size = 0;
    }
    
    addAtHead(val) {
        const newNode = new DoublyListNode(val);
        newNode.next = this.head.next;
        newNode.prev = this.head;
        this.head.next.prev = newNode;
        this.head.next = newNode;
        this.size++;
    }
    
    addAtTail(val) {
        const newNode = new DoublyListNode(val);
        newNode.next = this.tail;
        newNode.prev = this.tail.prev;
        this.tail.prev.next = newNode;
        this.tail.prev = newNode;
        this.size++;
    }
    
    deleteAtIndex(index) {
        if (index < 0 || index >= this.size) return;
        
        let current = this.head.next;
        for (let i = 0; i < index; i++) {
            current = current.next;
        }
        
        current.prev.next = current.next;
        current.next.prev = current.prev;
        this.size--;
    }
}
```

---

## 📚 **Stacks and Queues**

### **Stack Implementation**

```javascript
// Stack with Array
class Stack {
    constructor() {
        this.items = [];
    }
    
    push(item) {
        this.items.push(item);
    }
    
    pop() {
        if (this.isEmpty()) return null;
        return this.items.pop();
    }
    
    peek() {
        if (this.isEmpty()) return null;
        return this.items[this.items.length - 1];
    }
    
    isEmpty() {
        return this.items.length === 0;
    }
    
    size() {
        return this.items.length;
    }
}

// Stack with Linked List
class StackNode {
    constructor(val) {
        this.val = val;
        this.next = null;
    }
}

class LinkedStack {
    constructor() {
        this.top = null;
        this.size = 0;
    }
    
    push(val) {
        const newNode = new StackNode(val);
        newNode.next = this.top;
        this.top = newNode;
        this.size++;
    }
    
    pop() {
        if (this.isEmpty()) return null;
        
        const val = this.top.val;
        this.top = this.top.next;
        this.size--;
        return val;
    }
    
    peek() {
        return this.isEmpty() ? null : this.top.val;
    }
    
    isEmpty() {
        return this.top === null;
    }
}

// Queue Implementation
class Queue {
    constructor() {
        this.items = [];
    }
    
    enqueue(item) {
        this.items.push(item);
    }
    
    dequeue() {
        if (this.isEmpty()) return null;
        return this.items.shift();
    }
    
    front() {
        if (this.isEmpty()) return null;
        return this.items[0];
    }
    
    isEmpty() {
        return this.items.length === 0;
    }
    
    size() {
        return this.items.length;
    }
}

// Circular Queue
class CircularQueue {
    constructor(k) {
        this.queue = new Array(k);
        this.head = 0;
        this.tail = 0;
        this.size = 0;
        this.capacity = k;
    }
    
    enQueue(value) {
        if (this.isFull()) return false;
        
        this.queue[this.tail] = value;
        this.tail = (this.tail + 1) % this.capacity;
        this.size++;
        return true;
    }
    
    deQueue() {
        if (this.isEmpty()) return false;
        
        this.head = (this.head + 1) % this.capacity;
        this.size--;
        return true;
    }
    
    Front() {
        return this.isEmpty() ? -1 : this.queue[this.head];
    }
    
    Rear() {
        return this.isEmpty() ? -1 : this.queue[(this.tail - 1 + this.capacity) % this.capacity];
    }
    
    isEmpty() {
        return this.size === 0;
    }
    
    isFull() {
        return this.size === this.capacity;
    }
}

// Stack-based Problems
class StackProblems {
    // Valid Parentheses
    static isValid(s) {
        const stack = [];
        const mapping = {
            ')': '(',
            '}': '{',
            ']': '['
        };
        
        for (const char of s) {
            if (char in mapping) {
                if (stack.length === 0 || stack.pop() !== mapping[char]) {
                    return false;
                }
            } else {
                stack.push(char);
            }
        }
        
        return stack.length === 0;
    }
    
    // Daily Temperatures
    static dailyTemperatures(temperatures) {
        const result = new Array(temperatures.length).fill(0);
        const stack = [];
        
        for (let i = 0; i < temperatures.length; i++) {
            while (stack.length > 0 && temperatures[i] > temperatures[stack[stack.length - 1]]) {
                const index = stack.pop();
                result[index] = i - index;
            }
            stack.push(i);
        }
        
        return result;
    }
    
    // Largest Rectangle in Histogram
    static largestRectangleArea(heights) {
        const stack = [];
        let maxArea = 0;
        
        for (let i = 0; i <= heights.length; i++) {
            const h = i === heights.length ? 0 : heights[i];
            
            while (stack.length > 0 && h < heights[stack[stack.length - 1]]) {
                const height = heights[stack.pop()];
                const width = stack.length === 0 ? i : i - stack[stack.length - 1] - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            
            stack.push(i);
        }
        
        return maxArea;
    }
}
```

---

## 🌳 **Trees and Graphs**

### **Tree Implementation**

```javascript
// Binary Tree Node
class TreeNode {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// Tree Traversals
class TreeTraversals {
    // Preorder Traversal
    static preorder(root) {
        const result = [];
        
        function traverse(node) {
            if (!node) return;
            result.push(node.val);
            traverse(node.left);
            traverse(node.right);
        }
        
        traverse(root);
        return result;
    }
    
    // Inorder Traversal
    static inorder(root) {
        const result = [];
        
        function traverse(node) {
            if (!node) return;
            traverse(node.left);
            result.push(node.val);
            traverse(node.right);
        }
        
        traverse(root);
        return result;
    }
    
    // Postorder Traversal
    static postorder(root) {
        const result = [];
        
        function traverse(node) {
            if (!node) return;
            traverse(node.left);
            traverse(node.right);
            result.push(node.val);
        }
        
        traverse(root);
        return result;
    }
    
    // Level Order Traversal
    static levelOrder(root) {
        if (!root) return [];
        
        const result = [];
        const queue = [root];
        
        while (queue.length > 0) {
            const levelSize = queue.length;
            const currentLevel = [];
            
            for (let i = 0; i < levelSize; i++) {
                const node = queue.shift();
                currentLevel.push(node.val);
                
                if (node.left) queue.push(node.left);
                if (node.right) queue.push(node.right);
            }
            
            result.push(currentLevel);
        }
        
        return result;
    }
}

// Binary Search Tree
class BinarySearchTree {
    constructor() {
        this.root = null;
    }
    
    insert(val) {
        const newNode = new TreeNode(val);
        
        if (!this.root) {
            this.root = newNode;
            return;
        }
        
        let current = this.root;
        while (true) {
            if (val < current.val) {
                if (!current.left) {
                    current.left = newNode;
                    break;
                }
                current = current.left;
            } else {
                if (!current.right) {
                    current.right = newNode;
                    break;
                }
                current = current.right;
            }
        }
    }
    
    search(val) {
        let current = this.root;
        
        while (current) {
            if (val === current.val) return true;
            if (val < current.val) {
                current = current.left;
            } else {
                current = current.right;
            }
        }
        
        return false;
    }
    
    delete(val) {
        this.root = this.deleteNode(this.root, val);
    }
    
    deleteNode(node, val) {
        if (!node) return null;
        
        if (val < node.val) {
            node.left = this.deleteNode(node.left, val);
        } else if (val > node.val) {
            node.right = this.deleteNode(node.right, val);
        } else {
            if (!node.left) return node.right;
            if (!node.right) return node.left;
            
            const minNode = this.findMin(node.right);
            node.val = minNode.val;
            node.right = this.deleteNode(node.right, minNode.val);
        }
        
        return node;
    }
    
    findMin(node) {
        while (node.left) {
            node = node.left;
        }
        return node;
    }
}

// Graph Implementation
class Graph {
    constructor() {
        this.adjacencyList = new Map();
    }
    
    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }
    
    addEdge(vertex1, vertex2) {
        this.adjacencyList.get(vertex1).push(vertex2);
        this.adjacencyList.get(vertex2).push(vertex1);
    }
    
    // Depth First Search
    dfs(start) {
        const visited = new Set();
        const result = [];
        
        const traverse = (vertex) => {
            if (visited.has(vertex)) return;
            
            visited.add(vertex);
            result.push(vertex);
            
            const neighbors = this.adjacencyList.get(vertex);
            for (const neighbor of neighbors) {
                traverse(neighbor);
            }
        };
        
        traverse(start);
        return result;
    }
    
    // Breadth First Search
    bfs(start) {
        const visited = new Set();
        const queue = [start];
        const result = [];
        
        visited.add(start);
        
        while (queue.length > 0) {
            const vertex = queue.shift();
            result.push(vertex);
            
            const neighbors = this.adjacencyList.get(vertex);
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            }
        }
        
        return result;
    }
    
    // Shortest Path (BFS)
    shortestPath(start, end) {
        const visited = new Set();
        const queue = [[start, [start]]];
        
        visited.add(start);
        
        while (queue.length > 0) {
            const [vertex, path] = queue.shift();
            
            if (vertex === end) {
                return path;
            }
            
            const neighbors = this.adjacencyList.get(vertex);
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push([neighbor, [...path, neighbor]]);
                }
            }
        }
        
        return null;
    }
}
```

---

## 🎯 **Dynamic Programming**

### **DP Fundamentals**

```javascript
// Dynamic Programming Problems
class DynamicProgramming {
    // Fibonacci with Memoization
    static fibonacci(n, memo = {}) {
        if (n in memo) return memo[n];
        if (n <= 2) return 1;
        
        memo[n] = this.fibonacci(n - 1, memo) + this.fibonacci(n - 2, memo);
        return memo[n];
    }
    
    // Climbing Stairs
    static climbStairs(n) {
        if (n <= 2) return n;
        
        let prev2 = 1;
        let prev1 = 2;
        
        for (let i = 3; i <= n; i++) {
            const current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }
    
    // House Robber
    static rob(nums) {
        if (nums.length === 0) return 0;
        if (nums.length === 1) return nums[0];
        
        let prev2 = nums[0];
        let prev1 = Math.max(nums[0], nums[1]);
        
        for (let i = 2; i < nums.length; i++) {
            const current = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }
    
    // Longest Common Subsequence
    static longestCommonSubsequence(text1, text2) {
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
    
    // Edit Distance
    static minDistance(word1, word2) {
        const m = word1.length;
        const n = word2.length;
        const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0));
        
        // Initialize base cases
        for (let i = 0; i <= m; i++) dp[i][0] = i;
        for (let j = 0; j <= n; j++) dp[0][j] = j;
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (word1[i - 1] === word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(
                        dp[i - 1][j],     // delete
                        dp[i][j - 1],     // insert
                        dp[i - 1][j - 1]  // replace
                    );
                }
            }
        }
        
        return dp[m][n];
    }
    
    // Coin Change
    static coinChange(coins, amount) {
        const dp = Array(amount + 1).fill(Infinity);
        dp[0] = 0;
        
        for (let i = 1; i <= amount; i++) {
            for (const coin of coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] === Infinity ? -1 : dp[amount];
    }
    
    // Longest Increasing Subsequence
    static lengthOfLIS(nums) {
        const dp = Array(nums.length).fill(1);
        
        for (let i = 1; i < nums.length; i++) {
            for (let j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        
        return Math.max(...dp);
    }
    
    // 0/1 Knapsack
    static knapsack(weights, values, capacity) {
        const n = weights.length;
        const dp = Array(n + 1).fill().map(() => Array(capacity + 1).fill(0));
        
        for (let i = 1; i <= n; i++) {
            for (let w = 1; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                        dp[i - 1][w],
                        dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        
        return dp[n][capacity];
    }
}
```

---

## 🔍 **Sorting and Searching**

### **Sorting Algorithms**

```javascript
// Sorting Algorithms
class SortingAlgorithms {
    // Quick Sort
    static quickSort(arr, low = 0, high = arr.length - 1) {
        if (low < high) {
            const pivotIndex = this.partition(arr, low, high);
            this.quickSort(arr, low, pivotIndex - 1);
            this.quickSort(arr, pivotIndex + 1, high);
        }
        return arr;
    }
    
    static partition(arr, low, high) {
        const pivot = arr[high];
        let i = low - 1;
        
        for (let j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
        }
        
        [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
        return i + 1;
    }
    
    // Merge Sort
    static mergeSort(arr) {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const left = this.mergeSort(arr.slice(0, mid));
        const right = this.mergeSort(arr.slice(mid));
        
        return this.merge(left, right);
    }
    
    static merge(left, right) {
        const result = [];
        let i = 0, j = 0;
        
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result.push(left[i]);
                i++;
            } else {
                result.push(right[j]);
                j++;
            }
        }
        
        return result.concat(left.slice(i)).concat(right.slice(j));
    }
    
    // Heap Sort
    static heapSort(arr) {
        const n = arr.length;
        
        // Build max heap
        for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
            this.heapify(arr, n, i);
        }
        
        // Extract elements from heap
        for (let i = n - 1; i > 0; i--) {
            [arr[0], arr[i]] = [arr[i], arr[0]];
            this.heapify(arr, i, 0);
        }
        
        return arr;
    }
    
    static heapify(arr, n, i) {
        let largest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest]) {
            largest = left;
        }
        
        if (right < n && arr[right] > arr[largest]) {
            largest = right;
        }
        
        if (largest !== i) {
            [arr[i], arr[largest]] = [arr[largest], arr[i]];
            this.heapify(arr, n, largest);
        }
    }
}

// Searching Algorithms
class SearchingAlgorithms {
    // Binary Search
    static binarySearch(arr, target) {
        let left = 0;
        let right = arr.length - 1;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            
            if (arr[mid] === target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }
    
    // Find First and Last Position
    static searchRange(nums, target) {
        const first = this.findFirst(nums, target);
        if (first === -1) return [-1, -1];
        
        const last = this.findLast(nums, target);
        return [first, last];
    }
    
    static findFirst(nums, target) {
        let left = 0;
        let right = nums.length - 1;
        let result = -1;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            
            if (nums[mid] === target) {
                result = mid;
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    static findLast(nums, target) {
        let left = 0;
        let right = nums.length - 1;
        let result = -1;
        
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            
            if (nums[mid] === target) {
                result = mid;
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
}
```

---

## 🎯 **Key Takeaways**

### **Time Complexity**
- Arrays: O(1) access, O(n) search
- Linked Lists: O(n) access, O(1) insertion/deletion
- Stacks/Queues: O(1) operations
- Trees: O(log n) balanced, O(n) unbalanced
- Graphs: O(V + E) traversal

### **Space Complexity**
- Recursive solutions: O(depth) call stack
- Iterative solutions: O(1) to O(n) extra space
- Dynamic Programming: O(n) to O(n²) space

### **Common Patterns**
- Two Pointers: Array problems
- Sliding Window: Subarray problems
- Fast/Slow Pointers: Cycle detection
- Merge Intervals: Overlapping ranges
- Top K Elements: Heap usage

### **Interview Tips**
- Always clarify the problem
- Start with brute force, then optimize
- Consider edge cases
- Explain your approach clearly
- Test with examples

---

**🎉 This comprehensive guide covers all essential DSA concepts with Node.js implementations!**


## Hash Tables And Sets

<!-- AUTO-GENERATED ANCHOR: originally referenced as #hash-tables-and-sets -->

Placeholder content. Please replace with proper section.


## Advanced Algorithms

<!-- AUTO-GENERATED ANCHOR: originally referenced as #advanced-algorithms -->

Placeholder content. Please replace with proper section.
