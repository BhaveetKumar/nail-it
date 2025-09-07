# Binary Tree Maximum Path Sum

### Problem
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

**Example:**
```
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
```

### Golang Solution

```go
func maxPathSum(root *TreeNode) int {
    maxSum := math.MinInt32
    
    var maxGain func(*TreeNode) int
    maxGain = func(node *TreeNode) int {
        if node == nil {
            return 0
        }
        
        // Max gain from left and right subtrees
        leftGain := max(maxGain(node.Left), 0)
        rightGain := max(maxGain(node.Right), 0)
        
        // Current path sum (node + left + right)
        currentPathSum := node.Val + leftGain + rightGain
        
        // Update global maximum
        maxSum = max(maxSum, currentPathSum)
        
        // Return max gain for parent (node + max of left/right)
        return node.Val + max(leftGain, rightGain)
    }
    
    maxGain(root)
    return maxSum
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Using Global Variable**
```go
var maxSum int

func maxPathSumGlobal(root *TreeNode) int {
    maxSum = math.MinInt32
    maxGain(root)
    return maxSum
}

func maxGain(node *TreeNode) int {
    if node == nil {
        return 0
    }
    
    leftGain := max(maxGain(node.Left), 0)
    rightGain := max(maxGain(node.Right), 0)
    
    currentPathSum := node.Val + leftGain + rightGain
    maxSum = max(maxSum, currentPathSum)
    
    return node.Val + max(leftGain, rightGain)
}
```

#### **Iterative Approach**
```go
func maxPathSumIterative(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    maxSum := math.MinInt32
    stack := []*TreeNode{}
    lastVisited := (*TreeNode)(nil)
    current := root
    
    for current != nil || len(stack) > 0 {
        if current != nil {
            stack = append(stack, current)
            current = current.Left
        } else {
            peek := stack[len(stack)-1]
            
            if peek.Right != nil && lastVisited != peek.Right {
                current = peek.Right
            } else {
                // Process node
                leftGain := 0
                rightGain := 0
                
                if peek.Left != nil {
                    leftGain = max(peek.Left.Val, 0)
                }
                if peek.Right != nil {
                    rightGain = max(peek.Right.Val, 0)
                }
                
                currentPathSum := peek.Val + leftGain + rightGain
                maxSum = max(maxSum, currentPathSum)
                
                peek.Val += max(leftGain, rightGain)
                
                lastVisited = peek
                stack = stack[:len(stack)-1]
            }
        }
    }
    
    return maxSum
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(h) where h is height of tree
