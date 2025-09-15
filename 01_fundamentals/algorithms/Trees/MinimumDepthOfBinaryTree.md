# Minimum Depth of Binary Tree

### Problem
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: 2

Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5
```

### Golang Solution

```go
func minDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    if root.Left == nil && root.Right == nil {
        return 1
    }
    
    if root.Left == nil {
        return minDepth(root.Right) + 1
    }
    
    if root.Right == nil {
        return minDepth(root.Left) + 1
    }
    
    return min(minDepth(root.Left), minDepth(root.Right)) + 1
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Iterative BFS**
```go
func minDepthBFS(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    queue := []*TreeNode{root}
    depth := 1
    
    for len(queue) > 0 {
        levelSize := len(queue)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            // Check if it's a leaf node
            if node.Left == nil && node.Right == nil {
                return depth
            }
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
        
        depth++
    }
    
    return depth
}
```

#### **Iterative DFS**
```go
func minDepthDFS(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    stack := []*TreeNode{root}
    depths := []int{1}
    minDepth := math.MaxInt32
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        depth := depths[len(depths)-1]
        stack = stack[:len(stack)-1]
        depths = depths[:len(depths)-1]
        
        if node.Left == nil && node.Right == nil {
            minDepth = min(minDepth, depth)
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
    
    return minDepth
}
```

#### **Return Path to Minimum Depth Leaf**
```go
func minDepthWithPath(root *TreeNode) (int, []int) {
    if root == nil {
        return 0, []int{}
    }
    
    var minPath []int
    minDepth := math.MaxInt32
    
    var dfs func(*TreeNode, []int, int)
    dfs = func(node *TreeNode, path []int, depth int) {
        if node == nil {
            return
        }
        
        currentPath := append(path, node.Val)
        
        if node.Left == nil && node.Right == nil {
            if depth < minDepth {
                minDepth = depth
                minPath = make([]int, len(currentPath))
                copy(minPath, currentPath)
            }
            return
        }
        
        dfs(node.Left, currentPath, depth+1)
        dfs(node.Right, currentPath, depth+1)
    }
    
    dfs(root, []int{}, 1)
    return minDepth, minPath
}
```

#### **Return All Minimum Depth Paths**
```go
func allMinDepthPaths(root *TreeNode) [][]int {
    if root == nil {
        return [][]int{}
    }
    
    var allPaths [][]int
    minDepth := math.MaxInt32
    
    var dfs func(*TreeNode, []int, int)
    dfs = func(node *TreeNode, path []int, depth int) {
        if node == nil {
            return
        }
        
        currentPath := append(path, node.Val)
        
        if node.Left == nil && node.Right == nil {
            if depth < minDepth {
                minDepth = depth
                allPaths = [][]int{currentPath}
            } else if depth == minDepth {
                allPaths = append(allPaths, currentPath)
            }
            return
        }
        
        dfs(node.Left, currentPath, depth+1)
        dfs(node.Right, currentPath, depth+1)
    }
    
    dfs(root, []int{}, 1)
    return allPaths
}
```

#### **Return Depth Statistics**
```go
type MinDepthStats struct {
    MinDepth    int
    MaxDepth    int
    MinDepthCount int
    AllDepths   []int
}

func minDepthStatistics(root *TreeNode) MinDepthStats {
    if root == nil {
        return MinDepthStats{}
    }
    
    var allDepths []int
    minDepth := math.MaxInt32
    maxDepth := 0
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, depth int) {
        if node == nil {
            return
        }
        
        if node.Left == nil && node.Right == nil {
            allDepths = append(allDepths, depth)
            if depth < minDepth {
                minDepth = depth
            }
            if depth > maxDepth {
                maxDepth = depth
            }
            return
        }
        
        dfs(node.Left, depth+1)
        dfs(node.Right, depth+1)
    }
    
    dfs(root, 1)
    
    minDepthCount := 0
    for _, depth := range allDepths {
        if depth == minDepth {
            minDepthCount++
        }
    }
    
    return MinDepthStats{
        MinDepth:      minDepth,
        MaxDepth:      maxDepth,
        MinDepthCount: minDepthCount,
        AllDepths:     allDepths,
    }
}
```

#### **Check if Tree is Complete**
```go
func isCompleteTree(root *TreeNode) bool {
    if root == nil {
        return true
    }
    
    queue := []*TreeNode{root}
    foundNull := false
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        if node == nil {
            foundNull = true
        } else {
            if foundNull {
                return false
            }
            queue = append(queue, node.Left, node.Right)
        }
    }
    
    return true
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree
