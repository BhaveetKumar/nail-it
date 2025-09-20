# Trees Pattern

> **Master tree algorithms and traversals with Go implementations**

## 📋 Problems

### **Binary Tree Traversals**

- [Binary Tree Inorder Traversal](BinaryTreeInorderTraversal.md) - Left, Root, Right
- [Binary Tree Preorder Traversal](BinaryTreePreorderTraversal.md) - Root, Left, Right
- [Binary Tree Postorder Traversal](BinaryTreePostorderTraversal.md) - Left, Right, Root
- [Binary Tree Level Order Traversal](BinaryTreeLevelOrderTraversal.md) - BFS traversal
- [Binary Tree Zigzag Level Order Traversal](BinaryTreeZigzagLevelOrderTraversal.md) - Alternating directions

### **Binary Search Tree**

- [Validate Binary Search Tree](ValidateBinarySearchTree.md) - BST property validation
- [Search in a Binary Search Tree](SearchInBinarySearchTree.md) - BST search
- [Insert into a Binary Search Tree](InsertIntoBinarySearchTree.md) - BST insertion
- [Delete Node in a BST](DeleteNodeInBST.md) - BST deletion
- [Kth Smallest Element in a BST](KthSmallestElementInBST.md) - Inorder traversal

### **Tree Construction**

- [Construct Binary Tree from Preorder and Inorder Traversal](ConstructBinaryTreeFromPreorderAndInorder.md) - Tree reconstruction
- [Construct Binary Tree from Inorder and Postorder Traversal](ConstructBinaryTreeFromInorderAndPostorder.md) - Tree reconstruction
- [Serialize and Deserialize Binary Tree](SerializeAndDeserializeBinaryTree.md) - Tree serialization
- [Populating Next Right Pointers in Each Node](PopulatingNextRightPointers.md) - Level connection

### **Tree Properties**

- [Maximum Depth of Binary Tree](MaximumDepthOfBinaryTree.md) - Tree height
- [Minimum Depth of Binary Tree](MinimumDepthOfBinaryTree.md) - Minimum path length
- [Balanced Binary Tree](BalancedBinaryTree.md) - Height balance check
- [Symmetric Tree](SymmetricTree.md) - Mirror symmetry
- [Same Tree](SameTree.md) - Tree equality

### **Advanced Tree Problems**

- [Binary Tree Maximum Path Sum](BinaryTreeMaximumPathSum.md) - Path optimization
- [Lowest Common Ancestor of a Binary Tree](LowestCommonAncestor.md) - LCA algorithm
- [Path Sum](PathSum.md) - Target sum in paths
- [Path Sum II](PathSumII.md) - All paths with target sum
- [Binary Tree Right Side View](BinaryTreeRightSideView.md) - Rightmost nodes

---

## 🎯 Key Concepts

### **Tree Representation in Go**

**Detailed Explanation:**
Trees are hierarchical data structures where each node has at most one parent and zero or more children. In Go, trees are typically represented using structs with pointer fields to child nodes, making them perfect for recursive algorithms and memory-efficient storage.

**Binary Tree Structure:**

- **Node**: Contains data and pointers to left and right children
- **Root**: The topmost node with no parent
- **Leaf**: Nodes with no children
- **Internal Node**: Nodes with at least one child
- **Subtree**: Any node and its descendants form a subtree

**Go Implementation Benefits:**

- **Memory Efficiency**: Only allocates memory for existing nodes
- **Type Safety**: Compile-time type checking for tree operations
- **Garbage Collection**: Automatic memory management for tree nodes
- **Pointer Semantics**: Natural representation of parent-child relationships

```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Helper function to create a new node
func NewTreeNode(val int) *TreeNode {
    return &TreeNode{
        Val:   val,
        Left:  nil,
        Right: nil,
    }
}
```

**Extended Tree Node (for more complex trees):**

```go
type TreeNode struct {
    Val     int
    Left    *TreeNode
    Right   *TreeNode
    Parent  *TreeNode  // For upward traversal
    Next    *TreeNode  // For level connections
    Count   int        // For duplicate values
}
```

**Discussion Questions & Answers:**

**Q1: What are the advantages and disadvantages of different tree representations in Go?**

**Answer:** Representation trade-offs:

- **Pointer-based (current)**:
  - **Advantages**: Memory efficient, natural recursive structure, easy to implement
  - **Disadvantages**: No random access, harder to serialize, potential memory fragmentation
- **Array-based**:
  - **Advantages**: Better cache locality, easier serialization, random access
  - **Disadvantages**: Memory waste for sparse trees, fixed size, complex indexing
- **Parent pointers**:
  - **Advantages**: Easy upward traversal, useful for LCA problems
  - **Disadvantages**: Extra memory overhead, more complex maintenance

**Q2: How do you handle memory management for large trees in Go?**

**Answer:** Memory management strategies:

- **Garbage Collection**: Go's GC handles most memory management automatically
- **Object Pooling**: Reuse tree nodes for frequently created/destroyed trees
- **Lazy Allocation**: Only allocate child nodes when needed
- **Memory Profiling**: Use Go's memory profiler to identify memory leaks
- **Reference Counting**: For shared tree structures, implement reference counting
- **Weak References**: Use weak references for parent pointers to avoid cycles

### **Traversal Patterns**

**Detailed Explanation:**
Tree traversal is the process of visiting each node in a tree exactly once. Different traversal orders serve different purposes and are fundamental to many tree algorithms. Understanding when to use each traversal is crucial for solving tree problems efficiently.

**Inorder Traversal (Left → Root → Right):**

- **Use Case**: Binary Search Trees (gives sorted order), expression evaluation
- **Algorithm**: Recursively traverse left subtree, visit root, recursively traverse right subtree
- **Time Complexity**: O(n) where n is the number of nodes
- **Space Complexity**: O(h) where h is the height of the tree (recursion stack)
- **Key Insight**: For BST, inorder traversal gives elements in sorted order

**Preorder Traversal (Root → Left → Right):**

- **Use Case**: Tree reconstruction, expression tree evaluation, directory structure display
- **Algorithm**: Visit root, recursively traverse left subtree, recursively traverse right subtree
- **Time Complexity**: O(n)
- **Space Complexity**: O(h)
- **Key Insight**: Root is processed first, useful for tree serialization and reconstruction

**Postorder Traversal (Left → Right → Root):**

- **Use Case**: Tree deletion, expression evaluation, calculating directory sizes
- **Algorithm**: Recursively traverse left subtree, recursively traverse right subtree, visit root
- **Time Complexity**: O(n)
- **Space Complexity**: O(h)
- **Key Insight**: Root is processed last, useful when you need to process children before parent

**Level Order Traversal (Breadth-First):**

- **Use Case**: Level-based problems, finding shortest path, printing tree by levels
- **Algorithm**: Use queue to process nodes level by level
- **Time Complexity**: O(n)
- **Space Complexity**: O(w) where w is the maximum width of the tree
- **Key Insight**: Processes all nodes at the same level before moving to the next level

**Discussion Questions & Answers:**

**Q1: When should you use recursive vs iterative traversal implementations?**

**Answer:** Choose based on requirements:

- **Recursive**:
  - **Advantages**: Cleaner code, easier to understand, natural for tree problems
  - **Disadvantages**: Stack overflow risk for deep trees, function call overhead
  - **Best for**: Shallow trees, when code clarity is important
- **Iterative**:
  - **Advantages**: No stack overflow risk, better performance, more control
  - **Disadvantages**: More complex code, requires manual stack management
  - **Best for**: Deep trees, performance-critical applications, when avoiding recursion

**Q2: How do you implement iterative tree traversals efficiently in Go?**

**Answer:** Efficient iterative implementations:

- **Use Slices as Stacks**: Go slices are efficient for stack operations
- **Pre-allocate Capacity**: Use `make([]*TreeNode, 0, expectedSize)` for better performance
- **Avoid Repeated Allocations**: Reuse slices when possible
- **Use Pointers**: Work with pointers to avoid copying large structs
- **Early Termination**: Return early when possible to avoid unnecessary processing
- **Memory Management**: Clear slices when done to help garbage collection

### **Tree Properties**

**Detailed Explanation:**
Understanding tree properties is essential for analyzing tree algorithms and choosing appropriate data structures. These properties affect the performance and behavior of tree operations.

**Height and Depth:**

- **Height**: Maximum number of edges from root to any leaf
- **Depth**: Number of edges from root to a specific node
- **Relationship**: Height = maximum depth of any node
- **Calculation**: Height = max(height(left), height(right)) + 1
- **Use Cases**: Balancing algorithms, space complexity analysis, performance optimization

**Balanced Trees:**

- **Definition**: Height difference between left and right subtrees ≤ 1 for all nodes
- **Types**: AVL trees, Red-Black trees, B-trees
- **Benefits**: O(log n) height guarantees, predictable performance
- **Trade-offs**: More complex insertion/deletion, extra storage for balance information
- **Use Cases**: When you need guaranteed O(log n) operations

**Complete Trees:**

- **Definition**: All levels are completely filled except possibly the last level, which is filled from left to right
- **Properties**: Can be efficiently stored in arrays, parent-child relationships can be calculated
- **Use Cases**: Heaps, priority queues, efficient array-based tree representation
- **Benefits**: Better cache locality, easier serialization

**Full Trees:**

- **Definition**: Every node has either 0 or 2 children (no nodes with exactly 1 child)
- **Properties**: Number of leaves = number of internal nodes + 1
- **Use Cases**: Expression trees, decision trees, Huffman coding
- **Benefits**: Predictable structure, easier to analyze

**Discussion Questions & Answers:**

**Q1: How do you determine if a tree is balanced efficiently?**

**Answer:** Efficient balancing check:

- **Bottom-up Approach**: Calculate height of each subtree and check balance
- **Early Termination**: Return -1 if subtree is unbalanced to avoid further calculation
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(h) - recursion stack depth
- **Optimization**: Use iterative approach for very deep trees
- **Caching**: Cache height calculations if checking balance multiple times

**Q2: What are the performance implications of different tree properties?**

**Answer:** Performance characteristics:

- **Balanced Trees**: O(log n) operations, but O(log n) insertion/deletion due to rebalancing
- **Unbalanced Trees**: O(n) worst-case operations, but O(1) insertion/deletion
- **Complete Trees**: O(log n) operations, efficient array representation
- **Full Trees**: Predictable structure, easier to optimize
- **Trade-offs**: Balance maintenance vs operation speed, memory usage vs performance
- **Real-world**: Most applications use self-balancing trees for predictable performance

---

## 🛠️ Go-Specific Tips

### **Tree Traversal Implementations**

```go
// Recursive Inorder Traversal
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

// Iterative Inorder Traversal
func inorderTraversalIterative(root *TreeNode) []int {
    var result []int
    stack := []*TreeNode{}
    current := root

    for current != nil || len(stack) > 0 {
        // Go to the leftmost node
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
```

### **Level Order Traversal**

```go
func levelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }

    var result [][]int
    queue := []*TreeNode{root}

    for len(queue) > 0 {
        levelSize := len(queue)
        level := make([]int, 0, levelSize)

        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]

            level = append(level, node.Val)

            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }

        result = append(result, level)
    }

    return result
}
```

### **Tree Construction**

```go
func buildTree(preorder, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }

    // Create map for O(1) lookup
    inorderMap := make(map[int]int)
    for i, val := range inorder {
        inorderMap[val] = i
    }

    var build func(int, int) *TreeNode
    preIndex := 0

    build = func(left, right int) *TreeNode {
        if left > right {
            return nil
        }

        rootVal := preorder[preIndex]
        preIndex++

        root := &TreeNode{Val: rootVal}
        rootIndex := inorderMap[rootVal]

        root.Left = build(left, rootIndex-1)
        root.Right = build(rootIndex+1, right)

        return root
    }

    return build(0, len(inorder)-1)
}
```

### **Tree Validation**

```go
func isValidBST(root *TreeNode) bool {
    var validate func(*TreeNode, int, int) bool

    validate = func(node *TreeNode, min, max int) bool {
        if node == nil {
            return true
        }

        if node.Val <= min || node.Val >= max {
            return false
        }

        return validate(node.Left, min, node.Val) &&
               validate(node.Right, node.Val, max)
    }

    return validate(root, math.MinInt64, math.MaxInt64)
}
```

---

## 🎯 Interview Tips

### **How to Identify Tree Problems**

1. **Traversal Problems**: Use appropriate traversal (inorder, preorder, postorder, level order)
2. **BST Problems**: Use BST properties (left < root < right)
3. **Path Problems**: Use DFS with backtracking
4. **Level Problems**: Use BFS (level order traversal)
5. **Construction Problems**: Use recursive approach with proper indexing

### **Common Tree Problem Patterns**

- **Path Sum**: DFS with target sum
- **LCA (Lowest Common Ancestor)**: Use recursive approach
- **Tree Serialization**: Use preorder traversal with null markers
- **Tree Validation**: Use recursive validation with bounds
- **Tree Construction**: Use recursive construction with proper indexing

### **Optimization Tips**

- **Use iterative approach**: Avoid recursion stack overflow
- **Pre-allocate slices**: When size is known
- **Use maps for lookup**: O(1) access instead of O(n) search
- **Early termination**: Return as soon as condition is met
