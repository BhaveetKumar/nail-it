---
# Auto-generated front matter
Title: Maximumdepthofbinarytree
LastUpdated: 2025-11-06T20:45:58.696570
Tags: []
Status: draft
---

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

### Golang Solution

```go
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftDepth := maxDepth(root.Left)
    rightDepth := maxDepth(root.Right)
    
    return max(leftDepth, rightDepth) + 1
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Iterative BFS**
```go
func maxDepthBFS(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    queue := []*TreeNode{root}
    depth := 0
    
    for len(queue) > 0 {
        levelSize := len(queue)
        depth++
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
    }
    
    return depth
}
```

#### **Iterative DFS**
```go
func maxDepthDFS(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    stack := []*TreeNode{root}
    depths := []int{1}
    maxDepth := 0
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        depth := depths[len(depths)-1]
        stack = stack[:len(stack)-1]
        depths = depths[:len(depths)-1]
        
        if node.Left == nil && node.Right == nil {
            maxDepth = max(maxDepth, depth)
        }
        
        if node.Left != nil {
            stack = append(stack, node.Left)
            depths = append(depths, depth+1)
        }
        
        if node.Right != nil {
            stack = append(stack, node.Right)
            depths = append(depths, depth+1)
        }
    }
    
    return maxDepth
}
```

#### **Return Depth of Each Node**
```go
func maxDepthWithNodeDepths(root *TreeNode) (int, map[*TreeNode]int) {
    nodeDepths := make(map[*TreeNode]int)
    
    var dfs func(*TreeNode) int
    dfs = func(node *TreeNode) int {
        if node == nil {
            return 0
        }
        
        leftDepth := dfs(node.Left)
        rightDepth := dfs(node.Right)
        depth := max(leftDepth, rightDepth) + 1
        
        nodeDepths[node] = depth
        return depth
    }
    
    maxDepth := dfs(root)
    return maxDepth, nodeDepths
}
```

#### **Return All Depths**
```go
func allDepths(root *TreeNode) []int {
    var depths []int
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, depth int) {
        if node == nil {
            return
        }
        
        if node.Left == nil && node.Right == nil {
            depths = append(depths, depth)
        }
        
        dfs(node.Left, depth+1)
        dfs(node.Right, depth+1)
    }
    
    dfs(root, 1)
    return depths
}
```

#### **Return Depth Statistics**
```go
type DepthStats struct {
    MaxDepth    int
    MinDepth    int
    AvgDepth    float64
    TotalNodes  int
    LeafNodes   int
}

func depthStatistics(root *TreeNode) DepthStats {
    if root == nil {
        return DepthStats{}
    }
    
    var depths []int
    totalNodes := 0
    leafNodes := 0
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, depth int) {
        if node == nil {
            return
        }
        
        totalNodes++
        
        if node.Left == nil && node.Right == nil {
            depths = append(depths, depth)
            leafNodes++
        }
        
        dfs(node.Left, depth+1)
        dfs(node.Right, depth+1)
    }
    
    dfs(root, 1)
    
    if len(depths) == 0 {
        return DepthStats{MaxDepth: 1, MinDepth: 1, AvgDepth: 1, TotalNodes: 1, LeafNodes: 1}
    }
    
    maxDepth := depths[0]
    minDepth := depths[0]
    sum := 0
    
    for _, depth := range depths {
        if depth > maxDepth {
            maxDepth = depth
        }
        if depth < minDepth {
            minDepth = depth
        }
        sum += depth
    }
    
    return DepthStats{
        MaxDepth:   maxDepth,
        MinDepth:   minDepth,
        AvgDepth:   float64(sum) / float64(len(depths)),
        TotalNodes: totalNodes,
        LeafNodes:  leafNodes,
    }
}
```

#### **Check if Balanced**
```go
func isBalanced(root *TreeNode) bool {
    return checkHeight(root) != -1
}

func checkHeight(node *TreeNode) int {
    if node == nil {
        return 0
    }
    
    leftHeight := checkHeight(node.Left)
    if leftHeight == -1 {
        return -1
    }
    
    rightHeight := checkHeight(node.Right)
    if rightHeight == -1 {
        return -1
    }
    
    if abs(leftHeight-rightHeight) > 1 {
        return -1
    }
    
    return max(leftHeight, rightHeight) + 1
}

func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree