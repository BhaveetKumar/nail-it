---
# Auto-generated front matter
Title: Binarytreerightsideview
LastUpdated: 2025-11-06T20:45:58.696901
Tags: []
Status: draft
---

# Binary Tree Right Side View

### Problem
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

**Example:**
```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Input: root = [1,null,3]
Output: [1,3]
```

### Golang Solution

```go
func rightSideView(root *TreeNode) []int {
    if root == nil {
        return []int{}
    }
    
    var result []int
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        levelSize := len(queue)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            // Add the rightmost node of each level
            if i == levelSize-1 {
                result = append(result, node.Val)
            }
            
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
    }
    
    return result
}
```

### Alternative Solutions

#### **DFS Approach**
```go
func rightSideViewDFS(root *TreeNode) []int {
    var result []int
    dfs(root, 0, &result)
    return result
}

func dfs(node *TreeNode, level int, result *[]int) {
    if node == nil {
        return
    }
    
    // If this is the first node at this level, add it
    if level == len(*result) {
        *result = append(*result, node.Val)
    }
    
    // Visit right subtree first, then left
    dfs(node.Right, level+1, result)
    dfs(node.Left, level+1, result)
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(w) where w is maximum width
