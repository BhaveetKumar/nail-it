# Symmetric Tree

### Problem
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

**Example:**
```
Input: root = [1,2,2,3,4,4,3]
Output: true

Input: root = [1,2,2,null,3,null,3]
Output: false
```

### Golang Solution

```go
func isSymmetric(root *TreeNode) bool {
    if root == nil {
        return true
    }
    return isMirror(root.Left, root.Right)
}

func isMirror(left, right *TreeNode) bool {
    if left == nil && right == nil {
        return true
    }
    if left == nil || right == nil {
        return false
    }
    return left.Val == right.Val && isMirror(left.Left, right.Right) && isMirror(left.Right, right.Left)
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height
