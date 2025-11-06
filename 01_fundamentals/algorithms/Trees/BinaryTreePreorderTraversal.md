---
# Auto-generated front matter
Title: Binarytreepreordertraversal
LastUpdated: 2025-11-06T20:45:58.698502
Tags: []
Status: draft
---

# Binary Tree Preorder Traversal

### Problem
Given the root of a binary tree, return the preorder traversal of its nodes' values.

**Example:**
```
Input: root = [1,null,2,3]
Output: [1,2,3]

Input: root = []
Output: []

Input: root = [1]
Output: [1]
```

### Golang Solution

```go
func preorderTraversal(root *TreeNode) []int {
    var result []int
    
    var preorder func(*TreeNode)
    preorder = func(node *TreeNode) {
        if node == nil {
            return
        }
        
        result = append(result, node.Val)
        preorder(node.Left)
        preorder(node.Right)
    }
    
    preorder(root)
    return result
}
```

### Alternative Solutions

#### **Iterative with Stack**
```go
func preorderTraversalIterative(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    
    var result []int
    stack := []*TreeNode{root}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        result = append(result, node.Val)
        
        // Push right first, then left (so left is processed first)
        if node.Right != nil {
            stack = append(stack, node.Right)
        }
        if node.Left != nil {
            stack = append(stack, node.Left)
        }
    }
    
    return result
}
```

#### **Morris Traversal**
```go
func preorderTraversalMorris(root *TreeNode) []int {
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
                result = append(result, current.Val)
                predecessor.Right = current
                current = current.Left
            } else {
                predecessor.Right = nil
                current = current.Right
            }
        }
    }
    
    return result
}
```

#### **Return with Node Info**
```go
type PreorderNode struct {
    Value int
    Level int
    Index int
}

func preorderTraversalWithInfo(root *TreeNode) []PreorderNode {
    var result []PreorderNode
    index := 0
    
    var preorder func(*TreeNode, int)
    preorder = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        result = append(result, PreorderNode{
            Value: node.Val,
            Level: level,
            Index: index,
        })
        index++
        
        preorder(node.Left, level+1)
        preorder(node.Right, level+1)
    }
    
    preorder(root, 0)
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

#### **Return Level-wise Preorder**
```go
func preorderTraversalByLevel(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    levelMap := make(map[int][]int)
    
    var preorder func(*TreeNode, int)
    preorder = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        levelMap[level] = append(levelMap[level], node.Val)
        preorder(node.Left, level+1)
        preorder(node.Right, level+1)
    }
    
    preorder(root, 0)
    
    var result [][]int
    for i := 0; i < len(levelMap); i++ {
        result = append(result, levelMap[i])
    }
    
    return result
}
```

#### **Return with Statistics**
```go
type PreorderStats struct {
    TotalNodes    int
    LeafNodes     int
    InternalNodes int
    MaxLevel      int
    MinValue      int
    MaxValue      int
    Sum           int
}

func preorderTraversalStats(root *TreeNode) PreorderStats {
    if root == nil {
        return PreorderStats{}
    }
    
    stats := PreorderStats{
        MinValue: math.MaxInt32,
        MaxValue: math.MinInt32,
    }
    
    var preorder func(*TreeNode, int)
    preorder = func(node *TreeNode, level int) {
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
        
        preorder(node.Left, level+1)
        preorder(node.Right, level+1)
    }
    
    preorder(root, 0)
    return stats
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree