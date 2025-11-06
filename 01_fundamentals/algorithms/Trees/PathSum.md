---
# Auto-generated front matter
Title: Pathsum
LastUpdated: 2025-11-06T20:45:58.696052
Tags: []
Status: draft
---

# Path Sum

### Problem
Given the root of a binary tree and an integer `targetSum`, return `true` if the tree has a root-to-leaf path such that adding up all the values along the path equals `targetSum`.

A leaf is a node with no children.

**Example:**
```
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.

Input: root = [1,2,3], targetSum = 5
Output: false
Explanation: There two root-to-leaf paths in the tree:
(1 --> 2): The sum is 3.
(1 --> 3): The sum is 4.
There is no root-to-leaf path with sum = 5.
```

### Golang Solution

```go
func hasPathSum(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    
    if root.Left == nil && root.Right == nil {
        return root.Val == targetSum
    }
    
    return hasPathSum(root.Left, targetSum-root.Val) || 
           hasPathSum(root.Right, targetSum-root.Val)
}
```

### Alternative Solutions

#### **Iterative DFS**
```go
func hasPathSumIterative(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    
    stack := []*TreeNode{root}
    sumStack := []int{root.Val}
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        currentSum := sumStack[len(sumStack)-1]
        sumStack = sumStack[:len(sumStack)-1]
        
        if node.Left == nil && node.Right == nil {
            if currentSum == targetSum {
                return true
            }
        }
        
        if node.Right != nil {
            stack = append(stack, node.Right)
            sumStack = append(sumStack, currentSum+node.Right.Val)
        }
        
        if node.Left != nil {
            stack = append(stack, node.Left)
            sumStack = append(sumStack, currentSum+node.Left.Val)
        }
    }
    
    return false
}
```

#### **BFS Approach**
```go
func hasPathSumBFS(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    
    queue := []*TreeNode{root}
    sumQueue := []int{root.Val}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        currentSum := sumQueue[0]
        sumQueue = sumQueue[1:]
        
        if node.Left == nil && node.Right == nil {
            if currentSum == targetSum {
                return true
            }
        }
        
        if node.Left != nil {
            queue = append(queue, node.Left)
            sumQueue = append(sumQueue, currentSum+node.Left.Val)
        }
        
        if node.Right != nil {
            queue = append(queue, node.Right)
            sumQueue = append(sumQueue, currentSum+node.Right.Val)
        }
    }
    
    return false
}
```

#### **Return All Paths**
```go
func pathSum(root *TreeNode, targetSum int) [][]int {
    var result [][]int
    var currentPath []int
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, remainingSum int) {
        if node == nil {
            return
        }
        
        currentPath = append(currentPath, node.Val)
        
        if node.Left == nil && node.Right == nil && remainingSum == node.Val {
            path := make([]int, len(currentPath))
            copy(path, currentPath)
            result = append(result, path)
        }
        
        dfs(node.Left, remainingSum-node.Val)
        dfs(node.Right, remainingSum-node.Val)
        
        currentPath = currentPath[:len(currentPath)-1]
    }
    
    dfs(root, targetSum)
    return result
}
```

#### **Count Paths**
```go
func pathSumCount(root *TreeNode, targetSum int) int {
    if root == nil {
        return 0
    }
    
    return pathSumFromNode(root, targetSum) + 
           pathSumCount(root.Left, targetSum) + 
           pathSumCount(root.Right, targetSum)
}

func pathSumFromNode(node *TreeNode, targetSum int) int {
    if node == nil {
        return 0
    }
    
    count := 0
    if node.Val == targetSum {
        count = 1
    }
    
    return count + 
           pathSumFromNode(node.Left, targetSum-node.Val) + 
           pathSumFromNode(node.Right, targetSum-node.Val)
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree