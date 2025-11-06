---
# Auto-generated front matter
Title: Linkedlistoperations
LastUpdated: 2025-11-06T20:45:58.797366
Tags: []
Status: draft
---

# 🔗 Linked List Operations - Complete Guide

## Problem Statement

Implement and demonstrate various operations on linked lists including insertion, deletion, traversal, and advanced operations.

## Linked List Node Definition

```javascript
class ListNode {
  constructor(val = 0, next = null) {
    this.val = val;
    this.next = next;
  }
}
```

## Basic Operations

### 1. Create Linked List from Array

```javascript
/**
 * Create linked list from array
 * @param {number[]} arr
 * @return {ListNode}
 */
function createLinkedList(arr) {
  if (!arr || arr.length === 0) return null;
  
  const head = new ListNode(arr[0]);
  let current = head;
  
  for (let i = 1; i < arr.length; i++) {
    current.next = new ListNode(arr[i]);
    current = current.next;
  }
  
  return head;
}
```

### 2. Convert Linked List to Array

```javascript
/**
 * Convert linked list to array
 * @param {ListNode} head
 * @return {number[]}
 */
function linkedListToArray(head) {
  const result = [];
  let current = head;
  
  while (current) {
    result.push(current.val);
    current = current.next;
  }
  
  return result;
}
```

### 3. Print Linked List

```javascript
/**
 * Print linked list
 * @param {ListNode} head
 */
function printLinkedList(head) {
  const values = [];
  let current = head;
  
  while (current) {
    values.push(current.val);
    current = current.next;
  }
  
  console.log(values.join(" -> "));
}
```

## Insertion Operations

### 1. Insert at Beginning

```javascript
/**
 * Insert node at the beginning
 * @param {ListNode} head
 * @param {number} val
 * @return {ListNode}
 */
function insertAtBeginning(head, val) {
  const newNode = new ListNode(val);
  newNode.next = head;
  return newNode;
}
```

### 2. Insert at End

```javascript
/**
 * Insert node at the end
 * @param {ListNode} head
 * @param {number} val
 * @return {ListNode}
 */
function insertAtEnd(head, val) {
  const newNode = new ListNode(val);
  
  if (!head) return newNode;
  
  let current = head;
  while (current.next) {
    current = current.next;
  }
  
  current.next = newNode;
  return head;
}
```

### 3. Insert at Position

```javascript
/**
 * Insert node at specific position
 * @param {ListNode} head
 * @param {number} val
 * @param {number} position
 * @return {ListNode}
 */
function insertAtPosition(head, val, position) {
  const newNode = new ListNode(val);
  
  if (position === 0) {
    newNode.next = head;
    return newNode;
  }
  
  let current = head;
  for (let i = 0; i < position - 1 && current; i++) {
    current = current.next;
  }
  
  if (!current) return head; // Position out of bounds
  
  newNode.next = current.next;
  current.next = newNode;
  
  return head;
}
```

## Deletion Operations

### 1. Delete by Value

```javascript
/**
 * Delete first occurrence of value
 * @param {ListNode} head
 * @param {number} val
 * @return {ListNode}
 */
function deleteByValue(head, val) {
  if (!head) return null;
  
  if (head.val === val) {
    return head.next;
  }
  
  let current = head;
  while (current.next && current.next.val !== val) {
    current = current.next;
  }
  
  if (current.next) {
    current.next = current.next.next;
  }
  
  return head;
}
```

### 2. Delete at Position

```javascript
/**
 * Delete node at specific position
 * @param {ListNode} head
 * @param {number} position
 * @return {ListNode}
 */
function deleteAtPosition(head, position) {
  if (!head) return null;
  
  if (position === 0) {
    return head.next;
  }
  
  let current = head;
  for (let i = 0; i < position - 1 && current.next; i++) {
    current = current.next;
  }
  
  if (current.next) {
    current.next = current.next.next;
  }
  
  return head;
}
```

### 3. Delete All Occurrences

```javascript
/**
 * Delete all occurrences of value
 * @param {ListNode} head
 * @param {number} val
 * @return {ListNode}
 */
function deleteAllOccurrences(head, val) {
  // Remove nodes from beginning
  while (head && head.val === val) {
    head = head.next;
  }
  
  if (!head) return null;
  
  let current = head;
  while (current.next) {
    if (current.next.val === val) {
      current.next = current.next.next;
    } else {
      current = current.next;
    }
  }
  
  return head;
}
```

## Advanced Operations

### 1. Reverse Linked List

```javascript
/**
 * Reverse linked list iteratively
 * @param {ListNode} head
 * @return {ListNode}
 */
function reverseLinkedList(head) {
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

/**
 * Reverse linked list recursively
 * @param {ListNode} head
 * @return {ListNode}
 */
function reverseLinkedListRecursive(head) {
  if (!head || !head.next) return head;
  
  const newHead = reverseLinkedListRecursive(head.next);
  head.next.next = head;
  head.next = null;
  
  return newHead;
}
```

### 2. Find Middle Node

```javascript
/**
 * Find middle node using two pointers
 * @param {ListNode} head
 * @return {ListNode}
 */
function findMiddleNode(head) {
  if (!head) return null;
  
  let slow = head;
  let fast = head;
  
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  
  return slow;
}
```

### 3. Detect Cycle

```javascript
/**
 * Detect cycle in linked list
 * @param {ListNode} head
 * @return {boolean}
 */
function hasCycle(head) {
  if (!head || !head.next) return false;
  
  let slow = head;
  let fast = head;
  
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    
    if (slow === fast) return true;
  }
  
  return false;
}

/**
 * Find cycle start node
 * @param {ListNode} head
 * @return {ListNode}
 */
function detectCycleStart(head) {
  if (!head || !head.next) return null;
  
  let slow = head;
  let fast = head;
  
  // Find meeting point
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    
    if (slow === fast) break;
  }
  
  if (slow !== fast) return null; // No cycle
  
  // Find cycle start
  slow = head;
  while (slow !== fast) {
    slow = slow.next;
    fast = fast.next;
  }
  
  return slow;
}
```

### 4. Merge Two Sorted Lists

```javascript
/**
 * Merge two sorted linked lists
 * @param {ListNode} list1
 * @param {ListNode} list2
 * @return {ListNode}
 */
function mergeTwoLists(list1, list2) {
  const dummy = new ListNode(0);
  let current = dummy;
  
  while (list1 && list2) {
    if (list1.val <= list2.val) {
      current.next = list1;
      list1 = list1.next;
    } else {
      current.next = list2;
      list2 = list2.next;
    }
    current = current.next;
  }
  
  // Attach remaining nodes
  current.next = list1 || list2;
  
  return dummy.next;
}
```

### 5. Remove Nth Node from End

```javascript
/**
 * Remove nth node from end
 * @param {ListNode} head
 * @param {number} n
 * @return {ListNode}
 */
function removeNthFromEnd(head, n) {
  const dummy = new ListNode(0);
  dummy.next = head;
  
  let first = dummy;
  let second = dummy;
  
  // Move first pointer n+1 steps ahead
  for (let i = 0; i <= n; i++) {
    first = first.next;
  }
  
  // Move both pointers until first reaches end
  while (first) {
    first = first.next;
    second = second.next;
  }
  
  // Remove the nth node
  second.next = second.next.next;
  
  return dummy.next;
}
```

### 6. Palindrome Check

```javascript
/**
 * Check if linked list is palindrome
 * @param {ListNode} head
 * @return {boolean}
 */
function isPalindrome(head) {
  if (!head || !head.next) return true;
  
  // Find middle
  let slow = head;
  let fast = head;
  
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  
  // Reverse second half
  let secondHalf = reverseLinkedList(slow);
  let firstHalf = head;
  
  // Compare both halves
  while (secondHalf) {
    if (firstHalf.val !== secondHalf.val) {
      return false;
    }
    firstHalf = firstHalf.next;
    secondHalf = secondHalf.next;
  }
  
  return true;
}
```

### 7. Intersection of Two Lists

```javascript
/**
 * Find intersection of two linked lists
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
function getIntersectionNode(headA, headB) {
  if (!headA || !headB) return null;
  
  let a = headA;
  let b = headB;
  
  // When one pointer reaches end, move it to other list
  // This handles different lengths
  while (a !== b) {
    a = a ? a.next : headB;
    b = b ? b.next : headA;
  }
  
  return a;
}
```

## Doubly Linked List

```javascript
class DoublyListNode {
  constructor(val = 0, prev = null, next = null) {
    this.val = val;
    this.prev = prev;
    this.next = next;
  }
}

class DoublyLinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
    this.size = 0;
  }
  
  // Add to head
  addToHead(val) {
    const newNode = new DoublyListNode(val);
    
    if (!this.head) {
      this.head = this.tail = newNode;
    } else {
      newNode.next = this.head;
      this.head.prev = newNode;
      this.head = newNode;
    }
    
    this.size++;
  }
  
  // Add to tail
  addToTail(val) {
    const newNode = new DoublyListNode(val);
    
    if (!this.tail) {
      this.head = this.tail = newNode;
    } else {
      this.tail.next = newNode;
      newNode.prev = this.tail;
      this.tail = newNode;
    }
    
    this.size++;
  }
  
  // Remove from head
  removeFromHead() {
    if (!this.head) return null;
    
    const val = this.head.val;
    
    if (this.head === this.tail) {
      this.head = this.tail = null;
    } else {
      this.head = this.head.next;
      this.head.prev = null;
    }
    
    this.size--;
    return val;
  }
  
  // Remove from tail
  removeFromTail() {
    if (!this.tail) return null;
    
    const val = this.tail.val;
    
    if (this.head === this.tail) {
      this.head = this.tail = null;
    } else {
      this.tail = this.tail.prev;
      this.tail.next = null;
    }
    
    this.size--;
    return val;
  }
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Linked List Operations Test ===");

// Create linked list
const arr = [1, 2, 3, 4, 5];
let head = createLinkedList(arr);
console.log("Original list:");
printLinkedList(head);

// Insert operations
console.log("\n=== Insertion Operations ===");
head = insertAtBeginning(head, 0);
console.log("After inserting 0 at beginning:");
printLinkedList(head);

head = insertAtEnd(head, 6);
console.log("After inserting 6 at end:");
printLinkedList(head);

head = insertAtPosition(head, 99, 3);
console.log("After inserting 99 at position 3:");
printLinkedList(head);

// Delete operations
console.log("\n=== Deletion Operations ===");
head = deleteByValue(head, 99);
console.log("After deleting 99:");
printLinkedList(head);

head = deleteAtPosition(head, 0);
console.log("After deleting at position 0:");
printLinkedList(head);

// Advanced operations
console.log("\n=== Advanced Operations ===");
console.log("Middle node:", findMiddleNode(head).val);

head = reverseLinkedList(head);
console.log("After reversing:");
printLinkedList(head);

// Palindrome test
const palindromeList = createLinkedList([1, 2, 3, 2, 1]);
console.log("Palindrome test [1,2,3,2,1]:", isPalindrome(palindromeList));

// Cycle detection
const cycleList = createLinkedList([1, 2, 3, 4, 5]);
cycleList.next.next.next.next.next = cycleList.next; // Create cycle
console.log("Has cycle:", hasCycle(cycleList));

// Merge two sorted lists
const list1 = createLinkedList([1, 3, 5]);
const list2 = createLinkedList([2, 4, 6]);
const merged = mergeTwoLists(list1, list2);
console.log("Merged lists [1,3,5] and [2,4,6]:");
printLinkedList(merged);

// Doubly linked list
console.log("\n=== Doubly Linked List ===");
const dll = new DoublyLinkedList();
dll.addToHead(1);
dll.addToHead(2);
dll.addToTail(3);
console.log("Doubly linked list operations completed");
```

## Key Insights

1. **Two-pointer technique** is very useful for linked list problems
2. **Dummy nodes** help simplify edge cases
3. **Recursive solutions** are often cleaner but use more space
4. **Cycle detection** uses Floyd's algorithm (tortoise and hare)
5. **Reversing** can be done iteratively or recursively
6. **Merging** requires careful pointer management

## Common Mistakes

1. **Not handling null pointers** properly
2. **Losing reference** to head node
3. **Incorrect pointer updates** during operations
4. **Not considering edge cases** (empty list, single node)
5. **Memory leaks** in languages without garbage collection

## Related Problems

- [Reverse Linked List II](../../../algorithms/LinkedLists/ReverseLinkedListII.md)
- [Rotate List](../../../algorithms/LinkedLists/RotateList.md)
- [Swap Nodes in Pairs](../../../algorithms/LinkedLists/SwapNodesInPairs.md)
- [Copy List with Random Pointer](../../../algorithms/LinkedLists/CopyListWithRandomPointer.md)

## Interview Tips

1. **Always clarify** the problem requirements
2. **Handle edge cases** first
3. **Use dummy nodes** to simplify operations
4. **Explain your approach** before coding
5. **Test with examples** after implementation
6. **Discuss time/space complexity**
