---
# Auto-generated front matter
Title: Binarytreezigzaglevelordertraversal
LastUpdated: 2025-11-06T20:45:58.696736
Tags: []
Status: draft
---

# Binary Tree Zigzag Level Order Traversal

### Problem
Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Input: root = [1]
Output: [[1]]

Input: root = []
Output: []
```

### Golang Solution

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

### Alternative Solutions

#### **Using Two Stacks**
```go
func zigzagLevelOrderTwoStacks(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    currentLevel := []*TreeNode{root}
    nextLevel := []*TreeNode{}
    leftToRight := true
    
    for len(currentLevel) > 0 {
        level := make([]int, len(currentLevel))
        
        for i, node := range currentLevel {
            level[i] = node.Val
            
            if leftToRight {
                if node.Left != nil {
                    nextLevel = append([]*TreeNode{node.Left}, nextLevel...)
                }
                if node.Right != nil {
                    nextLevel = append([]*TreeNode{node.Right}, nextLevel...)
                }
            } else {
                if node.Right != nil {
                    nextLevel = append([]*TreeNode{node.Right}, nextLevel...)
                }
                if node.Left != nil {
                    nextLevel = append([]*TreeNode{node.Left}, nextLevel...)
                }
            }
        }
        
        result = append(result, level)
        currentLevel = nextLevel
        nextLevel = []*TreeNode{}
        leftToRight = !leftToRight
    }
    
    return result
}
```

#### **Recursive Approach**
```go
func zigzagLevelOrderRecursive(root *TreeNode) [][]int {
    var result [][]int
    
    var dfs func(*TreeNode, int, bool)
    dfs = func(node *TreeNode, level int, leftToRight bool) {
        if node == nil {
            return
        }
        
        if level >= len(result) {
            result = append(result, []int{})
        }
        
        if leftToRight {
            result[level] = append(result[level], node.Val)
        } else {
            result[level] = append([]int{node.Val}, result[level]...)
        }
        
        dfs(node.Left, level+1, !leftToRight)
        dfs(node.Right, level+1, !leftToRight)
    }
    
    dfs(root, 0, true)
    return result
}
```

#### **Using Deque**
```go
func zigzagLevelOrderDeque(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var result [][]int
    deque := []*TreeNode{root}
    leftToRight := true
    
    for len(deque) > 0 {
        levelSize := len(deque)
        level := make([]int, levelSize)
        
        for i := 0; i < levelSize; i++ {
            var node *TreeNode
            
            if leftToRight {
                node = deque[0]
                deque = deque[1:]
            } else {
                node = deque[len(deque)-1]
                deque = deque[:len(deque)-1]
            }
            
            level[i] = node.Val
            
            if leftToRight {
                if node.Left != nil {
                    deque = append(deque, node.Left)
                }
                if node.Right != nil {
                    deque = append(deque, node.Right)
                }
            } else {
                if node.Right != nil {
                    deque = append([]*TreeNode{node.Right}, deque...)
                }
                if node.Left != nil {
                    deque = append([]*TreeNode{node.Left}, deque...)
                }
            }
        }
        
        result = append(result, level)
        leftToRight = !leftToRight
    }
    
    return result
}
```

#### **Using Reverse Function**
```go
func zigzagLevelOrderReverse(root *TreeNode) [][]int {
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
            
            level[i] = node.Val
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        
        if !leftToRight {
            reverseSlice(level)
        }
        
        result = append(result, level)
        leftToRight = !leftToRight
    }
    
    return result
}

func reverseSlice(s []int) {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}
```

#### **Return with Direction Info**
```go
type LevelInfo struct {
    Values []int
    Direction string
}

func zigzagLevelOrderWithDirection(root *TreeNode) []LevelInfo {
    if root == nil {
        return []LevelInfo{}
    }
    
    var result []LevelInfo
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
        
        direction := "left-to-right"
        if !leftToRight {
            direction = "right-to-left"
        }
        
        result = append(result, LevelInfo{
            Values: level,
            Direction: direction,
        })
        
        leftToRight = !leftToRight
    }
    
    return result
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(w) where w is the maximum width of the tree