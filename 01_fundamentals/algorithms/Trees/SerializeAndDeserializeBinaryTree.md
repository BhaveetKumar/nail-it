# Serialize and Deserialize Binary Tree

### Problem
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

**Example:**
```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
```

### Golang Solution

```go
import (
    "fmt"
    "strconv"
    "strings"
)

type Codec struct{}

func Constructor() Codec {
    return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
    if root == nil {
        return "null"
    }
    
    left := this.serialize(root.Left)
    right := this.serialize(root.Right)
    
    return fmt.Sprintf("%d,%s,%s", root.Val, left, right)
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
    values := strings.Split(data, ",")
    index := 0
    
    var build func() *TreeNode
    build = func() *TreeNode {
        if index >= len(values) || values[index] == "null" {
            index++
            return nil
        }
        
        val, _ := strconv.Atoi(values[index])
        index++
        
        node := &TreeNode{Val: val}
        node.Left = build()
        node.Level = build()
        
        return node
    }
    
    return build()
}
```

### Alternative Solutions

#### **Level Order Serialization**
```go
type CodecLevelOrder struct{}

func NewCodecLevelOrder() *CodecLevelOrder {
    return &CodecLevelOrder{}
}

func (this *CodecLevelOrder) serialize(root *TreeNode) string {
    if root == nil {
        return "null"
    }
    
    var result []string
    queue := []*TreeNode{root}
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        if node == nil {
            result = append(result, "null")
        } else {
            result = append(result, strconv.Itoa(node.Val))
            queue = append(queue, node.Left, node.Right)
        }
    }
    
    return strings.Join(result, ",")
}

func (this *CodecLevelOrder) deserialize(data string) *TreeNode {
    values := strings.Split(data, ",")
    if len(values) == 0 || values[0] == "null" {
        return nil
    }
    
    val, _ := strconv.Atoi(values[0])
    root := &TreeNode{Val: val}
    queue := []*TreeNode{root}
    index := 1
    
    for len(queue) > 0 && index < len(values) {
        node := queue[0]
        queue = queue[1:]
        
        // Left child
        if index < len(values) && values[index] != "null" {
            val, _ := strconv.Atoi(values[index])
            node.Left = &TreeNode{Val: val}
            queue = append(queue, node.Left)
        }
        index++
        
        // Right child
        if index < len(values) && values[index] != "null" {
            val, _ := strconv.Atoi(values[index])
            node.Right = &TreeNode{Val: val}
            queue = append(queue, node.Right)
        }
        index++
    }
    
    return root
}
```

### Complexity
- **Time Complexity:** O(n) for both serialize and deserialize
- **Space Complexity:** O(n)
