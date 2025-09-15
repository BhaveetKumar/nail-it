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

### Golang Solution

```go
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
```

### Alternative Solutions

#### **Iterative with Stack**
```go
func inorderTraversalIterative(root *TreeNode) []int {
    var result []int
    var stack []*TreeNode
    current := root
    
    for current != nil || len(stack) > 0 {
        // Go to leftmost node
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

#### **Morris Traversal**
```go
func inorderTraversalMorris(root *TreeNode) []int {
    var result []int
    current := root
    
    for current != nil {
        if current.Left == nil {
            result = append(result, current.Val)
            current = current.Right
        } else {
            // Find inorder predecessor
            predecessor := current.Left
            for predecessor.Right != nil && predecessor.Right != current {
                predecessor = predecessor.Right
            }
            
            if predecessor.Right == nil {
                predecessor.Right = current
                current = current.Left
            } else {
                predecessor.Right = nil
                result = append(result, current.Val)
                current = current.Right
            }
        }
    }
    
    return result
}
```

#### **Return with Node Info**
```go
type TraversalNode struct {
    Value int
    Level int
    Index int
}

func inorderTraversalWithInfo(root *TreeNode) []TraversalNode {
    var result []TraversalNode
    index := 0
    
    var inorder func(*TreeNode, int)
    inorder = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        inorder(node.Left, level+1)
        result = append(result, TraversalNode{
            Value: node.Val,
            Level: level,
            Index: index,
        })
        index++
        inorder(node.Right, level+1)
    }
    
    inorder(root, 0)
    return result
}
```

#### **Return All Traversals**
```go
type AllTraversals struct {
    Inorder   []int
    Preorder  []int
    Postorder []int
}

func allTraversals(root *TreeNode) AllTraversals {
    var inorder, preorder, postorder []int
    
    var traverse func(*TreeNode)
    traverse = func(node *TreeNode) {
        if node == nil {
            return
        }
        
        preorder = append(preorder, node.Val)
        traverse(node.Left)
        inorder = append(inorder, node.Val)
        traverse(node.Right)
        postorder = append(postorder, node.Val)
    }
    
    traverse(root)
    
    return AllTraversals{
        Inorder:   inorder,
        Preorder:  preorder,
        Postorder: postorder,
    }
}
```

#### **Return Level-wise Inorder**
```go
func inorderTraversalByLevel(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    levelMap := make(map[int][]int)
    
    var inorder func(*TreeNode, int)
    inorder = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        inorder(node.Left, level+1)
        levelMap[level] = append(levelMap[level], node.Val)
        inorder(node.Right, level+1)
    }
    
    inorder(root, 0)
    
    var result [][]int
    for i := 0; i < len(levelMap); i++ {
        result = append(result, levelMap[i])
    }
    
    return result
}
```

#### **Return with Statistics**
```go
type TraversalStats struct {
    TotalNodes    int
    LeafNodes     int
    InternalNodes int
    MaxLevel      int
    MinValue      int
    MaxValue      int
    Sum           int
}

func inorderTraversalStats(root *TreeNode) TraversalStats {
    if root == nil {
        return TraversalStats{}
    }
    
    stats := TraversalStats{
        MinValue: math.MaxInt32,
        MaxValue: math.MinInt32,
    }
    
    var inorder func(*TreeNode, int)
    inorder = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        stats.TotalNodes++
        stats.Sum += node.Val
        
        if node.Val < stats.MinValue {
            stats.MinValue = node.Val
        }
        if node.Val > stats.MaxValue {
            stats.MaxValue = node.Val
        }
        
        if level > stats.MaxLevel {
            stats.MaxLevel = level
        }
        
        if node.Left == nil && node.Right == nil {
            stats.LeafNodes++
        } else {
            stats.InternalNodes++
        }
        
        inorder(node.Left, level+1)
        inorder(node.Right, level+1)
    }
    
    inorder(root, 0)
    return stats
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree