# 🌳 Binary Tree Traversal - Tree Problem

> **Master all types of binary tree traversals with iterative and recursive approaches**

## 📚 Overview

Binary tree traversal is a fundamental concept in computer science. This guide covers all major traversal methods: Pre-order, In-order, Post-order, and Level-order (BFS).

## 🎯 Tree Node Definition

```javascript
class TreeNode {
  constructor(val, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}

// Helper function to create a sample tree
function createSampleTree() {
  const root = new TreeNode(1);
  root.left = new TreeNode(2);
  root.right = new TreeNode(3);
  root.left.left = new TreeNode(4);
  root.left.right = new TreeNode(5);
  root.right.left = new TreeNode(6);
  root.right.right = new TreeNode(7);

  return root;
}

/*
Tree Structure:
       1
      / \
     2   3
    / \ / \
   4  5 6  7
*/
```

## 🔄 Traversal Methods

### **1. Pre-order Traversal (Root → Left → Right)**

```javascript
/**
 * Pre-order Traversal: Root → Left → Right
 * Time Complexity: O(n)
 * Space Complexity: O(h) where h is height of tree
 */

// Recursive Approach
function preorderRecursive(root) {
  const result = [];

  function traverse(node) {
    if (!node) return;

    result.push(node.val); // Visit root
    traverse(node.left); // Traverse left
    traverse(node.right); // Traverse right
  }

  traverse(root);
  return result;
}

// Iterative Approach using Stack
function preorderIterative(root) {
  if (!root) return [];

  const result = [];
  const stack = [root];

  while (stack.length > 0) {
    const node = stack.pop();
    result.push(node.val);

    // Push right first, then left (so left is processed first)
    if (node.right) stack.push(node.right);
    if (node.left) stack.push(node.left);
  }

  return result;
}

// Test
const tree = createSampleTree();
console.log("Pre-order Recursive:", preorderRecursive(tree)); // [1, 2, 4, 5, 3, 6, 7]
console.log("Pre-order Iterative:", preorderIterative(tree)); // [1, 2, 4, 5, 3, 6, 7]
```

### **2. In-order Traversal (Left → Root → Right)**

```javascript
/**
 * In-order Traversal: Left → Root → Right
 * For BST, this gives sorted order
 * Time Complexity: O(n)
 * Space Complexity: O(h)
 */

// Recursive Approach
function inorderRecursive(root) {
  const result = [];

  function traverse(node) {
    if (!node) return;

    traverse(node.left); // Traverse left
    result.push(node.val); // Visit root
    traverse(node.right); // Traverse right
  }

  traverse(root);
  return result;
}

// Iterative Approach using Stack
function inorderIterative(root) {
  const result = [];
  const stack = [];
  let current = root;

  while (current || stack.length > 0) {
    // Go to the leftmost node
    while (current) {
      stack.push(current);
      current = current.left;
    }

    // Process current node
    current = stack.pop();
    result.push(current.val);

    // Move to right subtree
    current = current.right;
  }

  return result;
}

// Test
console.log("In-order Recursive:", inorderRecursive(tree)); // [4, 2, 5, 1, 6, 3, 7]
console.log("In-order Iterative:", inorderIterative(tree)); // [4, 2, 5, 1, 6, 3, 7]
```

### **3. Post-order Traversal (Left → Right → Root)**

```javascript
/**
 * Post-order Traversal: Left → Right → Root
 * Useful for deleting trees, calculating directory sizes
 * Time Complexity: O(n)
 * Space Complexity: O(h)
 */

// Recursive Approach
function postorderRecursive(root) {
  const result = [];

  function traverse(node) {
    if (!node) return;

    traverse(node.left); // Traverse left
    traverse(node.right); // Traverse right
    result.push(node.val); // Visit root
  }

  traverse(root);
  return result;
}

// Iterative Approach using Two Stacks
function postorderIterative(root) {
  if (!root) return [];

  const result = [];
  const stack1 = [root];
  const stack2 = [];

  // First pass: push nodes to stack2 in reverse order
  while (stack1.length > 0) {
    const node = stack1.pop();
    stack2.push(node);

    if (node.left) stack1.push(node.left);
    if (node.right) stack1.push(node.right);
  }

  // Second pass: pop from stack2 to get post-order
  while (stack2.length > 0) {
    result.push(stack2.pop().val);
  }

  return result;
}

// Alternative Iterative Approach using One Stack
function postorderIterativeOneStack(root) {
  if (!root) return [];

  const result = [];
  const stack = [];
  let lastVisited = null;
  let current = root;

  while (current || stack.length > 0) {
    if (current) {
      stack.push(current);
      current = current.left;
    } else {
      const peekNode = stack[stack.length - 1];

      // If right child exists and hasn't been processed
      if (peekNode.right && lastVisited !== peekNode.right) {
        current = peekNode.right;
      } else {
        result.push(peekNode.val);
        lastVisited = stack.pop();
      }
    }
  }

  return result;
}

// Test
console.log("Post-order Recursive:", postorderRecursive(tree)); // [4, 5, 2, 6, 7, 3, 1]
console.log("Post-order Iterative:", postorderIterative(tree)); // [4, 5, 2, 6, 7, 3, 1]
console.log("Post-order One Stack:", postorderIterativeOneStack(tree)); // [4, 5, 2, 6, 7, 3, 1]
```

### **4. Level-order Traversal (BFS)**

```javascript
/**
 * Level-order Traversal (Breadth-First Search)
 * Process nodes level by level from left to right
 * Time Complexity: O(n)
 * Space Complexity: O(w) where w is maximum width
 */

// Basic Level-order
function levelOrder(root) {
  if (!root) return [];

  const result = [];
  const queue = [root];

  while (queue.length > 0) {
    const node = queue.shift();
    result.push(node.val);

    if (node.left) queue.push(node.left);
    if (node.right) queue.push(node.right);
  }

  return result;
}

// Level-order with Level Separation
function levelOrderWithLevels(root) {
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

// Level-order with Right View
function rightSideView(root) {
  if (!root) return [];

  const result = [];
  const queue = [root];

  while (queue.length > 0) {
    const levelSize = queue.length;

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift();

      // Add the last node of each level
      if (i === levelSize - 1) {
        result.push(node.val);
      }

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
  }

  return result;
}

// Test
console.log("Level-order:", levelOrder(tree)); // [1, 2, 3, 4, 5, 6, 7]
console.log("Level-order with Levels:", levelOrderWithLevels(tree)); // [[1], [2, 3], [4, 5, 6, 7]]
console.log("Right Side View:", rightSideView(tree)); // [1, 3, 7]
```

## 🔄 Advanced Traversal Patterns

### **1. Zigzag Level Order Traversal**

```javascript
/**
 * Zigzag Level Order Traversal
 * Alternate between left-to-right and right-to-left
 */
function zigzagLevelOrder(root) {
  if (!root) return [];

  const result = [];
  const queue = [root];
  let leftToRight = true;

  while (queue.length > 0) {
    const levelSize = queue.length;
    const currentLevel = [];

    for (let i = 0; i < levelSize; i++) {
      const node = queue.shift();

      if (leftToRight) {
        currentLevel.push(node.val);
      } else {
        currentLevel.unshift(node.val);
      }

      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }

    result.push(currentLevel);
    leftToRight = !leftToRight;
  }

  return result;
}

// Test
console.log("Zigzag Level Order:", zigzagLevelOrder(tree)); // [[1], [3, 2], [4, 5, 6, 7]]
```

### **2. Vertical Order Traversal**

```javascript
/**
 * Vertical Order Traversal
 * Group nodes by their horizontal distance from root
 */
function verticalOrder(root) {
  if (!root) return [];

  const columnTable = new Map();
  const queue = [[root, 0]]; // [node, column]

  while (queue.length > 0) {
    const [node, column] = queue.shift();

    if (!columnTable.has(column)) {
      columnTable.set(column, []);
    }
    columnTable.get(column).push(node.val);

    if (node.left) queue.push([node.left, column - 1]);
    if (node.right) queue.push([node.right, column + 1]);
  }

  // Sort columns and return values
  const sortedColumns = Array.from(columnTable.keys()).sort((a, b) => a - b);
  return sortedColumns.map((col) => columnTable.get(col));
}

// Test
console.log("Vertical Order:", verticalOrder(tree)); // [[4], [2], [1, 5, 6], [3], [7]]
```

### **3. Boundary Traversal**

```javascript
/**
 * Boundary Traversal
 * Traverse the boundary of the tree in anti-clockwise direction
 */
function boundaryTraversal(root) {
  if (!root) return [];

  const result = [];

  // Add root
  result.push(root.val);

  // Add left boundary (excluding leaves)
  addLeftBoundary(root.left, result);

  // Add leaves
  addLeaves(root, result);

  // Add right boundary (excluding leaves)
  addRightBoundary(root.right, result);

  return result;
}

function addLeftBoundary(node, result) {
  if (!node || (!node.left && !node.right)) return;

  result.push(node.val);

  if (node.left) {
    addLeftBoundary(node.left, result);
  } else {
    addLeftBoundary(node.right, result);
  }
}

function addLeaves(node, result) {
  if (!node) return;

  if (!node.left && !node.right) {
    result.push(node.val);
    return;
  }

  addLeaves(node.left, result);
  addLeaves(node.right, result);
}

function addRightBoundary(node, result) {
  if (!node || (!node.left && !node.right)) return;

  if (node.right) {
    addRightBoundary(node.right, result);
  } else {
    addRightBoundary(node.left, result);
  }

  result.push(node.val);
}

// Test
console.log("Boundary Traversal:", boundaryTraversal(tree)); // [1, 2, 4, 5, 6, 7, 3]
```

## 🧪 Comprehensive Test Suite

```javascript
// Test all traversal methods
function runTraversalTests() {
  const tree = createSampleTree();

  console.log("=== Binary Tree Traversal Tests ===\n");

  console.log("Tree Structure:");
  console.log("       1");
  console.log("      / \\");
  console.log("     2   3");
  console.log("    / \\ / \\");
  console.log("   4  5 6  7\n");

  console.log("Pre-order (Root → Left → Right):");
  console.log("Recursive:", preorderRecursive(tree));
  console.log("Iterative:", preorderIterative(tree));
  console.log();

  console.log("In-order (Left → Root → Right):");
  console.log("Recursive:", inorderRecursive(tree));
  console.log("Iterative:", inorderIterative(tree));
  console.log();

  console.log("Post-order (Left → Right → Root):");
  console.log("Recursive:", postorderRecursive(tree));
  console.log("Iterative:", postorderIterative(tree));
  console.log();

  console.log("Level-order (BFS):");
  console.log("Basic:", levelOrder(tree));
  console.log("With Levels:", levelOrderWithLevels(tree));
  console.log();

  console.log("Advanced Traversals:");
  console.log("Zigzag:", zigzagLevelOrder(tree));
  console.log("Vertical:", verticalOrder(tree));
  console.log("Boundary:", boundaryTraversal(tree));
}

runTraversalTests();
```

## 📊 Performance Comparison

```javascript
// Performance test for different traversal methods
function performanceTest() {
  // Create a large tree
  function createLargeTree(depth) {
    if (depth === 0) return null;

    const root = new TreeNode(Math.floor(Math.random() * 1000));
    root.left = createLargeTree(depth - 1);
    root.right = createLargeTree(depth - 1);

    return root;
  }

  const largeTree = createLargeTree(15); // ~32K nodes

  console.log("Performance Test with ~32K nodes:");

  console.time("Pre-order Recursive");
  preorderRecursive(largeTree);
  console.timeEnd("Pre-order Recursive");

  console.time("Pre-order Iterative");
  preorderIterative(largeTree);
  console.timeEnd("Pre-order Iterative");

  console.time("In-order Recursive");
  inorderRecursive(largeTree);
  console.timeEnd("In-order Recursive");

  console.time("In-order Iterative");
  inorderIterative(largeTree);
  console.timeEnd("In-order Iterative");

  console.time("Level-order");
  levelOrder(largeTree);
  console.timeEnd("Level-order");
}

// Uncomment to run performance test
// performanceTest();
```

## 🎯 Interview Tips

### **Key Points to Remember:**

1. **Pre-order**: Root first, useful for copying trees
2. **In-order**: Left first, gives sorted order for BST
3. **Post-order**: Root last, useful for deleting trees
4. **Level-order**: BFS, processes by levels

### **When to Use Each:**

- **Pre-order**: Tree serialization, expression trees
- **In-order**: BST operations, expression evaluation
- **Post-order**: Tree deletion, directory size calculation
- **Level-order**: Level-wise processing, shortest path

### **Common Follow-ups:**

1. Implement without recursion
2. Handle very deep trees (stack overflow)
3. Memory optimization
4. Custom traversal orders

---

**🎉 Master binary tree traversals to excel in tree-based problems!**

**Good luck with your coding interviews! 🚀**
