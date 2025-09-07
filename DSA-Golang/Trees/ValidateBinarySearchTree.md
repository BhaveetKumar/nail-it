# Validate Binary Search Tree

### Problem
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.

**Example:**
```
Input: root = [2,1,3]
Output: true

Input: root = [5,1,4,null,null,3,6]
Output: false
```

### Golang Solution

```go
func isValidBST(root *TreeNode) bool {
    return validate(root, nil, nil)
}

func validate(node *TreeNode, min, max *int) bool {
    if node == nil {
        return true
    }
    
    if (min != nil && node.Val <= *min) || (max != nil && node.Val >= *max) {
        return false
    }
    
    return validate(node.Left, min, &node.Val) && validate(node.Right, &node.Val, max)
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height
