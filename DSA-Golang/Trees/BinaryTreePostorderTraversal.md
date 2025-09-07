# Binary Tree Postorder Traversal

### Problem
Given the root of a binary tree, return the postorder traversal of its nodes' values.

**Example:**
```
Input: root = [1,null,2,3]
Output: [3,2,1]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

### Golang Solution

```go
// Recursive approach
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

// Iterative approach
func postorderTraversalIterative(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    
    var result []int
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        result = append(result, node.Val)
        
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
    }
    
    // Reverse the result
    for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
        result[i], result[j] = result[j], result[i]
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height
