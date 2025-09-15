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
    
    return isSymmetricHelper(root.Left, root.Right)
}

func isSymmetricHelper(left, right *TreeNode) bool {
    if left == nil && right == nil {
        return true
    }
    
    if left == nil || right == nil {
        return false
    }
    
    return left.Val == right.Val &&
           isSymmetricHelper(left.Left, right.Right) &&
           isSymmetricHelper(left.Right, right.Left)
}
```

### Alternative Solutions

#### **Iterative Using Queue**
```go
func isSymmetricIterative(root *TreeNode) bool {
    if root == nil {
        return true
    }
    
    queue := []*TreeNode{root.Left, root.Right}
    
    for len(queue) > 0 {
        left := queue[0]
        right := queue[1]
        queue = queue[2:]
        
        if left == nil && right == nil {
            continue
        }
        
        if left == nil || right == nil {
            return false
        }
        
        if left.Val != right.Val {
            return false
        }
        
        queue = append(queue, left.Left, right.Right, left.Right, right.Left)
    }
    
    return true
}
```

#### **Using Stack**
```go
func isSymmetricStack(root *TreeNode) bool {
    if root == nil {
        return true
    }
    
    stack := []*TreeNode{root.Left, root.Right}
    
    for len(stack) > 0 {
        right := stack[len(stack)-1]
        left := stack[len(stack)-2]
        stack = stack[:len(stack)-2]
        
        if left == nil && right == nil {
            continue
        }
        
        if left == nil || right == nil {
            return false
        }
        
        if left.Val != right.Val {
            return false
        }
        
        stack = append(stack, left.Left, right.Right, left.Right, right.Left)
    }
    
    return true
}
```

#### **Using Level Order Traversal**
```go
func isSymmetricLevelOrder(root *TreeNode) bool {
    if root == nil {
        return true
    }
    
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := make([]*int, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if node != nil {
                level[i] = &node.Val
                queue = append(queue, node.Left, node.Right)
            }
        }
        
        if !isLevelSymmetric(level) {
            return false
        }
    }
    
    return true
}

func isLevelSymmetric(level []*int) bool {
    left, right := 0, len(level)-1
    
    for left < right {
        if level[left] == nil && level[right] == nil {
            left++
            right--
            continue
        }
        
        if level[left] == nil || level[right] == nil {
            return false
        }
        
        if *level[left] != *level[right] {
            return false
        }
        
        left++
        right--
    }
    
    return true
}
```

#### **Return Symmetry Details**
```go
type SymmetryResult struct {
    IsSymmetric bool
    LeftCount   int
    RightCount  int
    Error       string
}

func isSymmetricWithDetails(root *TreeNode) SymmetryResult {
    if root == nil {
        return SymmetryResult{IsSymmetric: true}
    }
    
    leftCount := countNodes(root.Left)
    rightCount := countNodes(root.Right)
    
    if leftCount != rightCount {
        return SymmetryResult{
            IsSymmetric: false,
            LeftCount:   leftCount,
            RightCount:  rightCount,
            Error:       "Different number of nodes in left and right subtrees",
        }
    }
    
    isSymmetric := isSymmetricHelper(root.Left, root.Right)
    
    return SymmetryResult{
        IsSymmetric: isSymmetric,
        LeftCount:   leftCount,
        RightCount:  rightCount,
    }
}

func countNodes(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    return 1 + countNodes(root.Left) + countNodes(root.Right)
}
```

#### **Make Tree Symmetric**
```go
func makeSymmetric(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    // Create a symmetric copy
    return &TreeNode{
        Val:   root.Val,
        Left:  copyTree(root.Right),
        Right: copyTree(root.Left),
    }
}

func copyTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    return &TreeNode{
        Val:   root.Val,
        Left:  copyTree(root.Left),
        Right: copyTree(root.Right),
    }
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree