---
# Auto-generated front matter
Title: Constructbinarytreefrompreorderandinordertraversal
LastUpdated: 2025-11-06T20:45:58.695414
Tags: []
Status: draft
---

# Construct Binary Tree from Preorder and Inorder Traversal

### Problem
Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree.

**Example:**
```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

### Golang Solution

```go
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }
    
    // Create a map for O(1) lookup of inorder indices
    inorderMap := make(map[int]int)
    for i, val := range inorder {
        inorderMap[val] = i
    }
    
    preIndex := 0
    return buildTreeHelper(preorder, inorder, &preIndex, 0, len(inorder)-1, inorderMap)
}

func buildTreeHelper(preorder, inorder []int, preIndex *int, inStart, inEnd int, inorderMap map[int]int) *TreeNode {
    if inStart > inEnd {
        return nil
    }
    
    rootVal := preorder[*preIndex]
    *preIndex++
    
    root := &TreeNode{Val: rootVal}
    
    if inStart == inEnd {
        return root
    }
    
    inIndex := inorderMap[rootVal]
    
    root.Left = buildTreeHelper(preorder, inorder, preIndex, inStart, inIndex-1, inorderMap)
    root.Right = buildTreeHelper(preorder, inorder, preIndex, inIndex+1, inEnd, inorderMap)
    
    return root
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)


---

## AUTO-GENERATED: Starter Content

<!-- AUTO-GENERATED - REVIEW REQUIRED -->

This section seeds the document with a short introduction, learning objectives, and related links to code samples.

**Learning objectives:**
- Understand the core concepts.
- See practical code examples.

**Related files:**

Please replace this auto-generated section with curated content.
