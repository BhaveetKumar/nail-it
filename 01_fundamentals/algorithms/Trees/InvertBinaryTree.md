---
# Auto-generated front matter
Title: Invertbinarytree
LastUpdated: 2025-11-06T20:45:58.698042
Tags: []
Status: draft
---

# Invert Binary Tree

### Problem
Given the root of a binary tree, invert the tree, and return its root.

**Example:**
```
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Input: root = [2,1,3]
Output: [2,3,1]

Input: root = []
Output: []
```

### Golang Solution

```go
func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // Swap left and right subtrees
    root.Left, root.Right = root.Right, root.Left
    
    // Recursively invert subtrees
    invertTree(root.Left)
    invertTree(root.Right)
    
    return root
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
func invertTreeIterative(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        // Swap left and right
        node.Left, node.Right = node.Right, node.Left
        
        // Add children to queue
        if node.Left != nil {
            queue = append(queue, node.Left)
        }
        if node.Right != nil {
            queue = append(queue, node.Right)
        }
    }
    
    return root
}
```

#### **Using Stack**
```go
func invertTreeStack(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        // Swap left and right
        node.Left, node.Right = node.Right, node.Left
        
        // Add children to stack
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
    }
    
    return root
}
```

#### **Functional Approach**
```go
func invertTreeFunctional(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    return &TreeNode{
        Val:   root.Val,
        Left:  invertTreeFunctional(root.Right),
        Right: invertTreeFunctional(root.Left),
    }
}
```

#### **Post-order Traversal**
```go
func invertTreePostOrder(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // First invert the subtrees
    left := invertTreePostOrder(root.Left)
    right := invertTreePostOrder(root.Right)
    
    // Then swap them
    root.Left = right
    root.Right = left
    
    return root
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree
