# Flatten Binary Tree to Linked List

### Problem
Given the root of a binary tree, flatten the tree into a "linked list":

- The "linked list" should use the same `TreeNode` class where the `right` child pointer points to the next node in the list and the `left` child pointer is always `null`.
- The "linked list" should be in the same order as a pre-order traversal of the binary tree.

**Example:**
```
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
```

### Golang Solution

```go
func flatten(root *TreeNode) {
    if root == nil {
        return
    }
    
    // Flatten left and right subtrees
    flatten(root.Left)
    flatten(root.Right)
    
    // Store the right subtree
    right := root.Right
    
    // Move left subtree to right
    root.Right = root.Left
    root.Left = nil
    
    // Find the end of the new right subtree
    current := root
    for current.Right != nil {
        current = current.Right
    }
    
    // Attach the original right subtree
    current.Right = right
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
func flattenIterative(root *TreeNode) {
    if root == nil {
        return
    }
    
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if current.Right != nil {
            stack = append(stack, current.Right)
        }
        if current.Left != nil {
            stack = append(stack, current.Left)
        }
        
        if len(stack) > 0 {
            current.Right = stack[len(stack)-1]
        }
        current.Left = nil
    }
}
```

#### **Morris Traversal Style**
```go
func flattenMorris(root *TreeNode) {
    current := root
    
    for current != nil {
        if current.Left != nil {
            // Find the rightmost node in the left subtree
            rightmost := current.Left
            for rightmost.Right != nil {
                rightmost = rightmost.Right
            }
            
            // Make the right subtree the right child of rightmost
            rightmost.Right = current.Right
            
            // Move left subtree to right
            current.Right = current.Left
            current.Left = nil
        }
        
        current = current.Right
    }
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height of tree
