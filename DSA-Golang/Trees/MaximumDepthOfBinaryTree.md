# Maximum Depth of Binary Tree

### Problem
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: 3

Input: root = [1,null,2]
Output: 2
```

**Constraints:**
- The number of nodes in the tree is in the range [0, 10⁴]
- -100 ≤ Node.val ≤ 100

### Explanation

#### **Recursive Approach**
- Base case: if node is null, return 0
- Recursive case: return 1 + max(depth of left, depth of right)
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height

### Golang Solution

```go
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    return 1 + max(leftDepth, rightDepth)
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Notes / Variations

#### **Related Problems**
- **Minimum Depth of Binary Tree**: Find minimum depth
- **Balanced Binary Tree**: Check if tree is balanced
- **Diameter of Binary Tree**: Find longest path between nodes
- **Path Sum**: Check if path sum equals target
