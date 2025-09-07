# Binary Tree Preorder Traversal

### Problem
Given the root of a binary tree, return the preorder traversal of its nodes' values.

**Example:**
```
Input: root = [1,null,2,3]
Output: [1,2,3]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

### Golang Solution

```go
// Recursive approach
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

// Iterative approach
func preorderTraversalIterative(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    
    var result []int
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        result = append(result, node.Val)
        
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height
