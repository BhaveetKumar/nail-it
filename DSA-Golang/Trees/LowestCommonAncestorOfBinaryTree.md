# Lowest Common Ancestor of a Binary Tree

### Problem
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow a node to be a descendant of itself)."

**Example:**
```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5.
```

### Golang Solution

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    
    if left != nil && right != nil {
        return root
    }
    
    if left != nil {
        return left
    }
    
    return right
}
```

### Alternative Solutions

#### **Using Parent Pointers**
```go
func lowestCommonAncestorParent(root, p, q *TreeNode) *TreeNode {
    // Build parent map
    parent := make(map[*TreeNode]*TreeNode)
    buildParentMap(root, nil, parent)
    
    // Get path from p to root
    pathP := getPathToRoot(p, parent)
    
    // Find first common node in path from q to root
    current := q
    for current != nil {
        if contains(pathP, current) {
            return current
        }
        current = parent[current]
    }
    
    return nil
}

func buildParentMap(node, parentNode *TreeNode, parent map[*TreeNode]*TreeNode) {
    if node == nil {
        return
    }
    
    parent[node] = parentNode
    buildParentMap(node.Left, node, parent)
    buildParentMap(node.Right, node, parent)
}

func getPathToRoot(node *TreeNode, parent map[*TreeNode]*TreeNode) map[*TreeNode]bool {
    path := make(map[*TreeNode]bool)
    current := node
    
    for current != nil {
        path[current] = true
        current = parent[current]
    }
    
    return path
}

func contains(path map[*TreeNode]bool, node *TreeNode) bool {
    _, exists := path[node]
    return exists
}
```

#### **Iterative Approach**
```go
func lowestCommonAncestorIterative(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    
    stack := []*TreeNode{root}
    parent := make(map[*TreeNode]*TreeNode)
    parent[root] = nil
    
    // Build parent map using DFS
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        if node.Left != nil {
            parent[node.Left] = node
            stack = append(stack, node.Left)
        }
        
        if node.Right != nil {
            parent[node.Right] = node
            stack = append(stack, node.Right)
        }
    }
    
    // Get ancestors of p
    ancestors := make(map[*TreeNode]bool)
    current := p
    for current != nil {
        ancestors[current] = true
        current = parent[current]
    }
    
    // Find first common ancestor with q
    current = q
    for current != nil {
        if ancestors[current] {
            return current
        }
        current = parent[current]
    }
    
    return nil
}
```

#### **Using Path Arrays**
```go
func lowestCommonAncestorPath(root, p, q *TreeNode) *TreeNode {
    pathP := findPath(root, p)
    pathQ := findPath(root, q)
    
    if pathP == nil || pathQ == nil {
        return nil
    }
    
    // Find the last common node
    i := 0
    for i < len(pathP) && i < len(pathQ) && pathP[i] == pathQ[i] {
        i++
    }
    
    return pathP[i-1]
}

func findPath(root, target *TreeNode) []*TreeNode {
    if root == nil {
        return nil
    }
    
    if root == target {
        return []*TreeNode{root}
    }
    
    leftPath := findPath(root.Left, target)
    if leftPath != nil {
        return append([]*TreeNode{root}, leftPath...)
    }
    
    rightPath := findPath(root.Right, target)
    if rightPath != nil {
        return append([]*TreeNode{root}, rightPath...)
    }
    
    return nil
}
```

### Complexity
- **Time Complexity:** O(n) for all approaches
- **Space Complexity:** O(h) for recursive, O(n) for parent map approach
