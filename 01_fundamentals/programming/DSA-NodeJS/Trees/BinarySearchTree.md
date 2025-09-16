# 🌳 Binary Search Tree - Complete Implementation

## Problem Statement

Implement a complete Binary Search Tree (BST) with all essential operations including insertion, deletion, search, and various traversal methods.

## TreeNode Definition

```javascript
class TreeNode {
  constructor(val = 0, left = null, right = null) {
    this.val = val;
    this.left = left;
    this.right = right;
  }
}
```

## BST Class Implementation

```javascript
class BinarySearchTree {
  constructor() {
    this.root = null;
  }
  
  // Insert a value into BST
  insert(val) {
    this.root = this._insert(this.root, val);
  }
  
  _insert(node, val) {
    if (!node) {
      return new TreeNode(val);
    }
    
    if (val < node.val) {
      node.left = this._insert(node.left, val);
    } else if (val > node.val) {
      node.right = this._insert(node.right, val);
    }
    
    return node;
  }
  
  // Search for a value in BST
  search(val) {
    return this._search(this.root, val);
  }
  
  _search(node, val) {
    if (!node || node.val === val) {
      return node;
    }
    
    if (val < node.val) {
      return this._search(node.left, val);
    } else {
      return this._search(node.right, val);
    }
  }
  
  // Delete a value from BST
  delete(val) {
    this.root = this._delete(this.root, val);
  }
  
  _delete(node, val) {
    if (!node) return null;
    
    if (val < node.val) {
      node.left = this._delete(node.left, val);
    } else if (val > node.val) {
      node.right = this._delete(node.right, val);
    } else {
      // Node to be deleted found
      if (!node.left) return node.right;
      if (!node.right) return node.left;
      
      // Node with two children: get inorder successor
      const minNode = this._findMin(node.right);
      node.val = minNode.val;
      node.right = this._delete(node.right, minNode.val);
    }
    
    return node;
  }
  
  // Find minimum value node
  _findMin(node) {
    while (node.left) {
      node = node.left;
    }
    return node;
  }
  
  // Find maximum value node
  _findMax(node) {
    while (node.right) {
      node = node.right;
    }
    return node;
  }
  
  // Get height of tree
  height() {
    return this._height(this.root);
  }
  
  _height(node) {
    if (!node) return -1;
    return 1 + Math.max(this._height(node.left), this._height(node.right));
  }
  
  // Check if tree is valid BST
  isValidBST() {
    return this._isValidBST(this.root, -Infinity, Infinity);
  }
  
  _isValidBST(node, min, max) {
    if (!node) return true;
    
    if (node.val <= min || node.val >= max) return false;
    
    return this._isValidBST(node.left, min, node.val) &&
           this._isValidBST(node.right, node.val, max);
  }
}
```

## Traversal Methods

### 1. Inorder Traversal (Left-Root-Right)

```javascript
/**
 * Inorder traversal - gives sorted sequence for BST
 * @param {TreeNode} root
 * @return {number[]}
 */
function inorderTraversal(root) {
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

/**
 * Inorder traversal iterative
 * @param {TreeNode} root
 * @return {number[]}
 */
function inorderTraversalIterative(root) {
  const result = [];
  const stack = [];
  let current = root;
  
  while (current || stack.length > 0) {
    while (current) {
      stack.push(current);
      current = current.left;
    }
    
    current = stack.pop();
    result.push(current.val);
    current = current.right;
  }
  
  return result;
}
```

### 2. Preorder Traversal (Root-Left-Right)

```javascript
/**
 * Preorder traversal
 * @param {TreeNode} root
 * @return {number[]}
 */
function preorderTraversal(root) {
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

/**
 * Preorder traversal iterative
 * @param {TreeNode} root
 * @return {number[]}
 */
function preorderTraversalIterative(root) {
  if (!root) return [];
  
  const result = [];
  const stack = [root];
  
  while (stack.length > 0) {
    const node = stack.pop();
    result.push(node.val);
    
    if (node.right) stack.push(node.right);
    if (node.left) stack.push(node.left);
  }
  
  return result;
}
```

### 3. Postorder Traversal (Left-Right-Root)

```javascript
/**
 * Postorder traversal
 * @param {TreeNode} root
 * @return {number[]}
 */
function postorderTraversal(root) {
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

/**
 * Postorder traversal iterative
 * @param {TreeNode} root
 * @return {number[]}
 */
function postorderTraversalIterative(root) {
  if (!root) return [];
  
  const result = [];
  const stack = [root];
  
  while (stack.length > 0) {
    const node = stack.pop();
    result.unshift(node.val); // Add to beginning
    
    if (node.left) stack.push(node.left);
    if (node.right) stack.push(node.right);
  }
  
  return result;
}
```

### 4. Level Order Traversal (BFS)

```javascript
/**
 * Level order traversal (BFS)
 * @param {TreeNode} root
 * @return {number[][]}
 */
function levelOrderTraversal(root) {
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
```

## Advanced BST Operations

### 1. Find Kth Smallest Element

```javascript
/**
 * Find kth smallest element in BST
 * @param {TreeNode} root
 * @param {number} k
 * @return {number}
 */
function kthSmallest(root, k) {
  const stack = [];
  let current = root;
  let count = 0;
  
  while (current || stack.length > 0) {
    while (current) {
      stack.push(current);
      current = current.left;
    }
    
    current = stack.pop();
    count++;
    
    if (count === k) {
      return current.val;
    }
    
    current = current.right;
  }
  
  return -1;
}
```

### 2. Find Lowest Common Ancestor

```javascript
/**
 * Find lowest common ancestor in BST
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
function lowestCommonAncestor(root, p, q) {
  while (root) {
    if (p.val < root.val && q.val < root.val) {
      root = root.left;
    } else if (p.val > root.val && q.val > root.val) {
      root = root.right;
    } else {
      return root;
    }
  }
  
  return null;
}
```

### 3. Convert Sorted Array to BST

```javascript
/**
 * Convert sorted array to balanced BST
 * @param {number[]} nums
 * @return {TreeNode}
 */
function sortedArrayToBST(nums) {
  function buildBST(left, right) {
    if (left > right) return null;
    
    const mid = Math.floor((left + right) / 2);
    const root = new TreeNode(nums[mid]);
    
    root.left = buildBST(left, mid - 1);
    root.right = buildBST(mid + 1, right);
    
    return root;
  }
  
  return buildBST(0, nums.length - 1);
}
```

### 4. Validate BST

```javascript
/**
 * Validate if binary tree is BST
 * @param {TreeNode} root
 * @return {boolean}
 */
function isValidBST(root) {
  function validate(node, min, max) {
    if (!node) return true;
    
    if (node.val <= min || node.val >= max) return false;
    
    return validate(node.left, min, node.val) &&
           validate(node.right, node.val, max);
  }
  
  return validate(root, -Infinity, Infinity);
}
```

### 5. Recover BST

```javascript
/**
 * Recover BST with two swapped nodes
 * @param {TreeNode} root
 * @return {void}
 */
function recoverTree(root) {
  let first = null;
  let second = null;
  let prev = null;
  
  function inorder(node) {
    if (!node) return;
    
    inorder(node.left);
    
    if (prev && prev.val > node.val) {
      if (!first) {
        first = prev;
        second = node;
      } else {
        second = node;
      }
    }
    
    prev = node;
    inorder(node.right);
  }
  
  inorder(root);
  
  // Swap the values
  if (first && second) {
    const temp = first.val;
    first.val = second.val;
    second.val = temp;
  }
}
```

### 6. Range Sum of BST

```javascript
/**
 * Range sum of BST
 * @param {TreeNode} root
 * @param {number} low
 * @param {number} high
 * @return {number}
 */
function rangeSumBST(root, low, high) {
  if (!root) return 0;
  
  let sum = 0;
  
  if (root.val >= low && root.val <= high) {
    sum += root.val;
  }
  
  if (root.val > low) {
    sum += rangeSumBST(root.left, low, high);
  }
  
  if (root.val < high) {
    sum += rangeSumBST(root.right, low, high);
  }
  
  return sum;
}
```

## BST Iterator

```javascript
class BSTIterator {
  constructor(root) {
    this.stack = [];
    this._pushAll(root);
  }
  
  _pushAll(node) {
    while (node) {
      this.stack.push(node);
      node = node.left;
    }
  }
  
  next() {
    const node = this.stack.pop();
    this._pushAll(node.right);
    return node.val;
  }
  
  hasNext() {
    return this.stack.length > 0;
  }
}
```

## Test Cases

```javascript
// Test cases
console.log("=== Binary Search Tree Test ===");

// Create BST
const bst = new BinarySearchTree();
const values = [5, 3, 7, 2, 4, 6, 8];
values.forEach(val => bst.insert(val));

console.log("BST created with values:", values);
console.log("Height:", bst.height());
console.log("Is valid BST:", bst.isValidBST());

// Traversals
console.log("\n=== Traversals ===");
console.log("Inorder (recursive):", inorderTraversal(bst.root));
console.log("Inorder (iterative):", inorderTraversalIterative(bst.root));
console.log("Preorder (recursive):", preorderTraversal(bst.root));
console.log("Preorder (iterative):", preorderTraversalIterative(bst.root));
console.log("Postorder (recursive):", postorderTraversal(bst.root));
console.log("Postorder (iterative):", postorderTraversalIterative(bst.root));
console.log("Level order:", levelOrderTraversal(bst.root));

// Search operations
console.log("\n=== Search Operations ===");
console.log("Search 4:", bst.search(4) ? "Found" : "Not found");
console.log("Search 9:", bst.search(9) ? "Found" : "Not found");

// Advanced operations
console.log("\n=== Advanced Operations ===");
console.log("3rd smallest:", kthSmallest(bst.root, 3));
console.log("Range sum [3, 7]:", rangeSumBST(bst.root, 3, 7));

// BST Iterator
console.log("\n=== BST Iterator ===");
const iterator = new BSTIterator(bst.root);
const iteratorResult = [];
while (iterator.hasNext()) {
  iteratorResult.push(iterator.next());
}
console.log("Iterator result:", iteratorResult);

// Convert sorted array to BST
console.log("\n=== Convert Sorted Array to BST ===");
const sortedArray = [1, 2, 3, 4, 5, 6, 7];
const balancedBST = sortedArrayToBST(sortedArray);
console.log("Balanced BST from sorted array:");
console.log("Level order:", levelOrderTraversal(balancedBST));

// Delete operations
console.log("\n=== Delete Operations ===");
console.log("Before deletion:", inorderTraversal(bst.root));
bst.delete(3);
console.log("After deleting 3:", inorderTraversal(bst.root));
bst.delete(7);
console.log("After deleting 7:", inorderTraversal(bst.root));
```

## Visualization

```javascript
/**
 * Visualize BST structure
 * @param {TreeNode} root
 */
function visualizeBST(root) {
  function printTree(node, prefix = "", isLast = true) {
    if (!node) return;
    
    console.log(prefix + (isLast ? "└── " : "├── ") + node.val);
    
    const children = [];
    if (node.left) children.push(node.left);
    if (node.right) children.push(node.right);
    
    children.forEach((child, index) => {
      const isLastChild = index === children.length - 1;
      const newPrefix = prefix + (isLast ? "    " : "│   ");
      printTree(child, newPrefix, isLastChild);
    });
  }
  
  console.log("BST Structure:");
  printTree(root);
}

// Example visualization
console.log("=== BST Visualization ===");
visualizeBST(bst.root);
```

## Key Insights

1. **BST Property**: Left child < Parent < Right child
2. **Inorder Traversal**: Always gives sorted sequence
3. **Insertion**: Always at leaf position
4. **Deletion**: Three cases (no children, one child, two children)
5. **Search**: O(log n) average case, O(n) worst case
6. **Balanced BST**: Ensures O(log n) operations

## Common Mistakes

1. **Not maintaining BST property** during operations
2. **Incorrect deletion logic** for nodes with two children
3. **Not handling edge cases** (empty tree, single node)
4. **Memory leaks** in recursive implementations
5. **Incorrect traversal implementations**

## Related Problems

- [Validate Binary Search Tree](ValidateBinarySearchTree.md/)
- [Convert Sorted Array to BST](ConvertSortedArrayToBST.md/)
- [Kth Smallest Element in BST](KthSmallestElementInBST.md/)
- [Lowest Common Ancestor of BST](LowestCommonAncestorOfBST.md/)

## Interview Tips

1. **Understand BST properties** thoroughly
2. **Practice all traversal methods** (recursive and iterative)
3. **Handle edge cases** properly
4. **Explain time/space complexity** for each operation
5. **Discuss balancing** and its importance
6. **Know when to use BST** vs other data structures
