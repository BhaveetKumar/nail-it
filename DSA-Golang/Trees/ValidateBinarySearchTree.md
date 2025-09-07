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
Explanation: The root node's value is 5 but its right child's value is 4.
```

### Golang Solution

```go
func isValidBST(root *TreeNode) bool {
    return isValidBSTHelper(root, nil, nil)
}

func isValidBSTHelper(node *TreeNode, min, max *int) bool {
    if node == nil {
        return true
    }
    
    if min != nil && node.Val <= *min {
        return false
    }
    
    if max != nil && node.Val >= *max {
        return false
    }
    
    return isValidBSTHelper(node.Left, min, &node.Val) &&
           isValidBSTHelper(node.Right, &node.Val, max)
}
```

### Alternative Solutions

#### **Using Inorder Traversal**
```go
func isValidBSTInorder(root *TreeNode) bool {
    var values []int
    inorderTraversal(root, &values)
    
    for i := 1; i < len(values); i++ {
        if values[i] <= values[i-1] {
            return false
        }
    }
    
    return true
}

func inorderTraversal(node *TreeNode, values *[]int) {
    if node == nil {
        return
    }
    
    inorderTraversal(node.Left, values)
    *values = append(*values, node.Val)
    inorderTraversal(node.Right, values)
}
```

#### **Using Stack (Iterative Inorder)**
```go
func isValidBSTStack(root *TreeNode) bool {
    if root == nil {
        return true
    }
    
    stack := []*TreeNode{}
    var prev *int
    current := root
    
    for len(stack) > 0 || current != nil {
        for current != nil {
            stack = append(stack, current)
            current = current.Left
        }
        
        current = stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if prev != nil && current.Val <= *prev {
            return false
        }
        
        prev = &current.Val
        current = current.Right
    }
    
    return true
}
```

#### **Using Min/Max Values**
```go
func isValidBSTMinMax(root *TreeNode) bool {
    return isValidBSTMinMaxHelper(root, math.MinInt64, math.MaxInt64)
}

func isValidBSTMinMaxHelper(node *TreeNode, min, max int) bool {
    if node == nil {
        return true
    }
    
    if node.Val <= min || node.Val >= max {
        return false
    }
    
    return isValidBSTMinMaxHelper(node.Left, min, node.Val) &&
           isValidBSTMinMaxHelper(node.Right, node.Val, max)
}
```

#### **Return Validation Details**
```go
type BSTValidationResult struct {
    IsValid bool
    Min     int
    Max     int
    Error   string
}

func isValidBSTWithDetails(root *TreeNode) BSTValidationResult {
    if root == nil {
        return BSTValidationResult{IsValid: true, Min: 0, Max: 0}
    }
    
    result := validateBSTHelper(root)
    return result
}

func validateBSTHelper(node *TreeNode) BSTValidationResult {
    if node == nil {
        return BSTValidationResult{IsValid: true, Min: math.MaxInt64, Max: math.MinInt64}
    }
    
    left := validateBSTHelper(node.Left)
    right := validateBSTHelper(node.Right)
    
    if !left.IsValid || !right.IsValid {
        return BSTValidationResult{IsValid: false, Error: "Invalid subtree"}
    }
    
    if node.Left != nil && left.Max >= node.Val {
        return BSTValidationResult{IsValid: false, Error: "Left subtree max >= current"}
    }
    
    if node.Right != nil && right.Min <= node.Val {
        return BSTValidationResult{IsValid: false, Error: "Right subtree min <= current"}
    }
    
    minVal := node.Val
    if node.Left != nil {
        minVal = left.Min
    }
    
    maxVal := node.Val
    if node.Right != nil {
        maxVal = right.Max
    }
    
    return BSTValidationResult{IsValid: true, Min: minVal, Max: maxVal}
}
```

#### **Fix Invalid BST**
```go
func fixInvalidBST(root *TreeNode) *TreeNode {
    var first, second, prev *TreeNode
    
    var inorder func(*TreeNode)
    inorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        
        inorder(node.Left)
        
        if prev != nil && prev.Val > node.Val {
            if first == nil {
                first = prev
            }
            second = node
        }
        
        prev = node
        inorder(node.Right)
    }
    
    inorder(root)
    
    if first != nil && second != nil {
        first.Val, second.Val = second.Val, first.Val
    }
    
    return root
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree