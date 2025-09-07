# Trees Pattern

> **Master tree algorithms and traversals with Go implementations**

## 📋 Problems

### **Binary Tree Traversals**
- [Binary Tree Inorder Traversal](./BinaryTreeInorderTraversal.md) - Left, Root, Right
- [Binary Tree Preorder Traversal](./BinaryTreePreorderTraversal.md) - Root, Left, Right
- [Binary Tree Postorder Traversal](./BinaryTreePostorderTraversal.md) - Left, Right, Root
- [Binary Tree Level Order Traversal](./BinaryTreeLevelOrderTraversal.md) - BFS traversal
- [Binary Tree Zigzag Level Order Traversal](./BinaryTreeZigzagLevelOrderTraversal.md) - Alternating directions

### **Binary Search Tree**
- [Validate Binary Search Tree](./ValidateBinarySearchTree.md) - BST property validation
- [Search in a Binary Search Tree](./SearchInBinarySearchTree.md) - BST search
- [Insert into a Binary Search Tree](./InsertIntoBinarySearchTree.md) - BST insertion
- [Delete Node in a BST](./DeleteNodeInBST.md) - BST deletion
- [Kth Smallest Element in a BST](./KthSmallestElementInBST.md) - Inorder traversal

### **Tree Construction**
- [Construct Binary Tree from Preorder and Inorder Traversal](./ConstructBinaryTreeFromPreorderAndInorder.md) - Tree reconstruction
- [Construct Binary Tree from Inorder and Postorder Traversal](./ConstructBinaryTreeFromInorderAndPostorder.md) - Tree reconstruction
- [Serialize and Deserialize Binary Tree](./SerializeAndDeserializeBinaryTree.md) - Tree serialization
- [Populating Next Right Pointers in Each Node](./PopulatingNextRightPointers.md) - Level connection

### **Tree Properties**
- [Maximum Depth of Binary Tree](./MaximumDepthOfBinaryTree.md) - Tree height
- [Minimum Depth of Binary Tree](./MinimumDepthOfBinaryTree.md) - Minimum path length
- [Balanced Binary Tree](./BalancedBinaryTree.md) - Height balance check
- [Symmetric Tree](./SymmetricTree.md) - Mirror symmetry
- [Same Tree](./SameTree.md) - Tree equality

### **Advanced Tree Problems**
- [Binary Tree Maximum Path Sum](./BinaryTreeMaximumPathSum.md) - Path optimization
- [Lowest Common Ancestor of a Binary Tree](./LowestCommonAncestor.md) - LCA algorithm
- [Path Sum](./PathSum.md) - Target sum in paths
- [Path Sum II](./PathSumII.md) - All paths with target sum
- [Binary Tree Right Side View](./BinaryTreeRightSideView.md) - Rightmost nodes

---

## 🎯 Key Concepts

### **Tree Representation in Go**
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

### **Traversal Patterns**
- **Inorder**: Left → Root → Right (gives sorted order for BST)
- **Preorder**: Root → Left → Right (useful for tree reconstruction)
- **Postorder**: Left → Right → Root (useful for deletion)
- **Level Order**: Breadth-first traversal (useful for level-based problems)

### **Tree Properties**
- **Height**: Maximum depth from root to leaf
- **Depth**: Distance from root to a specific node
- **Balanced**: Height difference between left and right subtrees ≤ 1
- **Complete**: All levels filled except possibly the last level
- **Full**: Every node has either 0 or 2 children

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
