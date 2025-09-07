# Binary Tree Zigzag Level Order Traversal

### Problem
Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

**Example:**
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Input: root = [1]
Output: [[1]]
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
        level := make([]int, 0, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            if leftToRight {
                level = append(level, node.Val)
            } else {
                level = append([]int{node.Val}, level...)
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
- **Time Complexity:** O(n)
- **Space Complexity:** O(w) where w is maximum width
