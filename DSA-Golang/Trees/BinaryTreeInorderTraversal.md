# Binary Tree Inorder Traversal

### Problem
Given the root of a binary tree, return the inorder traversal of its nodes' values.

**Example:**
```
Input: root = [1,null,2,3]
Output: [1,3,2]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

**Constraints:**
- The number of nodes in the tree is in the range [0, 100]
- -100 ≤ Node.val ≤ 100

### Explanation

#### **Recursive Approach**
- Visit left subtree
- Process current node
- Visit right subtree
- Time Complexity: O(n)
- Space Complexity: O(h) where h is height

#### **Iterative Approach**
- Use stack to simulate recursion
- Push left nodes first, then process and move right
- Time Complexity: O(n)
- Space Complexity: O(h)

### Golang Solution

```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

// Recursive approach
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

// Iterative approach
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

### Notes / Variations

#### **Related Problems**
- **Binary Tree Preorder Traversal**: Root, Left, Right
- **Binary Tree Postorder Traversal**: Left, Right, Root
- **Binary Tree Level Order Traversal**: BFS traversal
- **Validate Binary Search Tree**: Check BST property
