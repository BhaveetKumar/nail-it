# Convert Sorted Array to Binary Search Tree

### Problem
Given an integer array `nums` where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

**Example:**
```
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:

Input: nums = [1,3]
Output: [3,1]
Explanation: [1,3] and [3,1] are both height-balanced BSTs.
```

### Golang Solution

```go
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    
    return buildBST(nums, 0, len(nums)-1)
}

func buildBST(nums []int, left, right int) *TreeNode {
    if left > right {
        return nil
    }
    
    mid := left + (right-left)/2
    
    root := &TreeNode{Val: nums[mid]}
    root.Left = buildBST(nums, left, mid-1)
    root.Right = buildBST(nums, mid+1, right)
    
    return root
}
```

### Alternative Solutions

#### **Iterative Approach**
```go
type NodeRange struct {
    node  *TreeNode
    left  int
    right int
}

func sortedArrayToBSTIterative(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    
    stack := []NodeRange{{node: &TreeNode{}, left: 0, right: len(nums) - 1}}
    
    for len(stack) > 0 {
        current := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        mid := current.left + (current.right-current.left)/2
        current.node.Val = nums[mid]
        
        if current.left <= mid-1 {
            current.node.Left = &TreeNode{}
            stack = append(stack, NodeRange{
                node:  current.node.Left,
                left:  current.left,
                right: mid - 1,
            })
        }
        
        if mid+1 <= current.right {
            current.node.Right = &TreeNode{}
            stack = append(stack, NodeRange{
                node:  current.node.Right,
                left:  mid + 1,
                right: current.right,
            })
        }
    }
    
    return stack[0].node
}
```

#### **Using Queue**
```go
type QueueNode struct {
    node  *TreeNode
    left  int
    right int
}

func sortedArrayToBSTQueue(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    
    root := &TreeNode{}
    queue := []QueueNode{{node: root, left: 0, right: len(nums) - 1}}
    
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        
        mid := current.left + (current.right-current.left)/2
        current.node.Val = nums[mid]
        
        if current.left <= mid-1 {
            current.node.Left = &TreeNode{}
            queue = append(queue, QueueNode{
                node:  current.node.Left,
                left:  current.left,
                right: mid - 1,
            })
        }
        
        if mid+1 <= current.right {
            current.node.Right = &TreeNode{}
            queue = append(queue, QueueNode{
                node:  current.node.Right,
                left:  mid + 1,
                right: current.right,
            })
        }
    }
    
    return root
}
```

### Complexity
- **Time Complexity:** O(n)
- **Space Complexity:** O(log n) for recursive, O(n) for iterative
