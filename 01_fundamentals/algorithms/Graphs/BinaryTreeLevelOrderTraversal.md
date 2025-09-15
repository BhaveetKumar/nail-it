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

**Constraints:**
- The number of nodes in the tree is in the range [0, 2000]
- -1000 ≤ Node.val ≤ 1000

### Explanation

#### **BFS Approach**
- Use queue to process nodes level by level
- For each level, process all nodes at that level
- Time Complexity: O(n)
- Space Complexity: O(w) where w is maximum width

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
        level := make([]int, 0, levelSize)
        
        for i := 0; i < levelSize; i++ {
            node := queue[0]
            queue = queue[1:]
            
            level = append(level, node.Val)
            
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

### Notes / Variations

#### **Related Problems**
- **Binary Tree Zigzag Level Order Traversal**: Alternate directions
- **Binary Tree Right Side View**: See rightmost nodes
- **Average of Levels in Binary Tree**: Calculate level averages
- **Minimum Depth of Binary Tree**: Find minimum depth
