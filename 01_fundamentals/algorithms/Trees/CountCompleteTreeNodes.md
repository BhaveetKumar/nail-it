---
# Auto-generated front matter
Title: Countcompletetreenodes
LastUpdated: 2025-11-06T20:45:58.698788
Tags: []
Status: draft
---

# Count Complete Tree Nodes

### Problem
Given the root of a complete binary tree, return the number of the nodes in the tree.

In a complete binary tree, every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

**Example:**
```
Input: root = [1,2,3,4,5,6]
Output: 6

Input: root = []
Output: 0

Input: root = [1]
Output: 1
```

### Golang Solution

```go
func countNodes(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftHeight := getLeftHeight(root)
    rightHeight := getRightHeight(root)
    
    if leftHeight == rightHeight {
        // Perfect binary tree
        return (1 << leftHeight) - 1
    }
    
    return 1 + countNodes(root.Left) + countNodes(root.Right)
}

func getLeftHeight(node *TreeNode) int {
    height := 0
    for node != nil {
        height++
        node = node.Left
    }
    return height
}

func getRightHeight(node *TreeNode) int {
    height := 0
    for node != nil {
        height++
        node = node.Right
    }
    return height
}
```

### Alternative Solutions

#### **Binary Search Approach**
```go
func countNodesBinarySearch(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    leftHeight := getLeftHeight(root)
    rightHeight := getRightHeight(root)
    
    if leftHeight == rightHeight {
        return (1 << leftHeight) - 1
    }
    
    // Binary search for the last level
    left, right := 0, (1<<(leftHeight-1))-1
    
    for left <= right {
        mid := left + (right-left)/2
        if exists(root, leftHeight, mid) {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return (1<<(leftHeight-1)) - 1 + left
}

func exists(node *TreeNode, height, index int) bool {
    left, right := 0, (1<<(height-1))-1
    
    for i := 0; i < height-1; i++ {
        mid := left + (right-left)/2
        if index <= mid {
            node = node.Left
            right = mid
        } else {
            node = node.Right
            left = mid + 1
        }
    }
    
    return node != nil
}
```

### Complexity
- **Time Complexity:** O(log² n)
- **Space Complexity:** O(log n)
