---
# Auto-generated front matter
Title: Binarytreelevelordertraversal
LastUpdated: 2025-11-06T20:45:58.694823
Tags: []
Status: draft
---

# Binary Tree Level Order Traversal

### Problem
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Input: root = [1]
Output: [[1]]

Input: root = []
Output: []
```

### Golang Solution

```go
func levelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := make([]int, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            level[i] = node.Val
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        
        result = append(result, level)
    }
    
    return result
}
```

### Alternative Solutions

#### **Recursive DFS**
```go
func levelOrderRecursive(root *TreeNode) [][]int {
    var result [][]int
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, level int) {
        if node == nil {
            return
        }
        
        if level >= len(result) {
            result = append(result, []int{})
        }
        
        result[level] = append(result[level], node.Val)
        
        dfs(node.Left, level+1)
        dfs(node.Right, level+1)
    }
    
    dfs(root, 0)
    return result
}
```

#### **Using Two Queues**
```go
func levelOrderTwoQueues(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    currentLevel := []*TreeNode{root}
    
    for len(currentLevel) > 0 {
        var nextLevel []*TreeNode
        level := make([]int, len(currentLevel))
        
        for i, node := range currentLevel {
            level[i] = node.Val
            
            if node.Left != nil {
                nextLevel = append(nextLevel, node.Left)
            }
            if node.Right != nil {
                nextLevel = append(nextLevel, node.Right)
            }
        }
        
        result = append(result, level)
        currentLevel = nextLevel
    }
    
    return result
}
```

#### **Using Stack (DFS)**
```go
func levelOrderStack(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    stack := []*TreeNode{root}
    levels := []int{0}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        level := levels[len(levels)-1]
        stack = stack[:len(stack)-1]
        levels = levels[:len(levels)-1]
        
        if level >= len(result) {
            result = append(result, []int{})
        }
        
        result[level] = append(result[level], node.Val)
        
        if node.Right != nil {
            stack = append(stack, node.Right)
            levels = append(levels, level+1)
        }
        
        if node.Left != nil {
            stack = append(stack, node.Left)
            levels = append(levels, level+1)
        }
    }
    
    return result
}
```

#### **Return with Node References**
```go
func levelOrderWithNodes(root *TreeNode) [][]*TreeNode {
    if root == nil {
        return [][]*TreeNode{}
    }
    
    var result [][]*TreeNode
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := make([]*TreeNode, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            level[i] = node
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        
        result = append(result, level)
    }
    
    return result
}
```

#### **Zigzag Level Order**
```go
func zigzagLevelOrder(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    queue := []*TreeNode{root}
    leftToRight := true
    
    for len(queue) > 0 {
        levelSize := len(queue)
        level := make([]int, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if leftToRight {
                level[i] = node.Val
            } else {
                level[levelSize-1-i] = node.Val
            }
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        
        result = append(result, level)
        leftToRight = !leftToRight
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(w) where w is the maximum width of the tree
