# Sum Root to Leaf Numbers

### Problem
You are given the root of a binary tree containing digits from `0` to `9` only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path `1 -> 2 -> 3` represents the number `123`.

Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.

**Example:**
```
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.
```

### Golang Solution

```go
func sumNumbers(root *TreeNode) int {
    return sumNumbersHelper(root, 0)
}

func sumNumbersHelper(node *TreeNode, currentSum int) int {
    if node == nil {
        return 0
    }
    
    currentSum = currentSum*10 + node.Val
    
    if node.Left == nil && node.Right == nil {
        return currentSum
    }
    
    return sumNumbersHelper(node.Left, currentSum) + 
           sumNumbersHelper(node.Right, currentSum)
}
```

### Alternative Solutions

#### **Iterative DFS**
```go
func sumNumbersIterative(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    stack := []*TreeNode{root}
    sumStack := []int{root.Val}
    totalSum := 0
    
    for len(stack) > 0 {
        node := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        
        currentSum := sumStack[len(sumStack)-1]
        sumStack = sumStack[:len(sumStack)-1]
        
        if node.Left == nil && node.Right == nil {
            totalSum += currentSum
        }
        
        if node.Right != nil {
            stack = append(stack, node.Right)
            sumStack = append(sumStack, currentSum*10+node.Right.Val)
        }
        
        if node.Left != nil {
            stack = append(stack, node.Left)
            sumStack = append(sumStack, currentSum*10+node.Left.Val)
        }
    }
    
    return totalSum
}
```

#### **BFS Approach**
```go
func sumNumbersBFS(root *TreeNode) int {
    if root == nil {
        return 0
    }
    
    queue := []*TreeNode{root}
    sumQueue := []int{root.Val}
    totalSum := 0
    
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        
        currentSum := sumQueue[0]
        sumQueue = sumQueue[1:]
        
        if node.Left == nil && node.Right == nil {
            totalSum += currentSum
        }
        
        if node.Left != nil {
            queue = append(queue, node.Left)
            sumQueue = append(sumQueue, currentSum*10+node.Left.Val)
        }
        
        if node.Right != nil {
            queue = append(queue, node.Right)
            sumQueue = append(sumQueue, currentSum*10+node.Right.Val)
        }
    }
    
    return totalSum
}
```

#### **Return All Numbers**
```go
func sumNumbersAll(root *TreeNode) (int, []int) {
    var allNumbers []int
    
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, currentSum int) {
        if node == nil {
            return
        }
        
        currentSum = currentSum*10 + node.Val
        
        if node.Left == nil && node.Right == nil {
            allNumbers = append(allNumbers, currentSum)
        }
        
        dfs(node.Left, currentSum)
        dfs(node.Right, currentSum)
    }
    
    dfs(root, 0)
    
    totalSum := 0
    for _, num := range allNumbers {
        totalSum += num
    }
    
    return totalSum, allNumbers
}
```

#### **Using String Conversion**
```go
func sumNumbersString(root *TreeNode) int {
    var allPaths []string
    
    var dfs func(*TreeNode, string)
    dfs = func(node *TreeNode, path string) {
        if node == nil {
            return
        }
        
        path += strconv.Itoa(node.Val)
        
        if node.Left == nil && node.Right == nil {
            allPaths = append(allPaths, path)
        }
        
        dfs(node.Left, path)
        dfs(node.Right, path)
    }
    
    dfs(root, "")
    
    totalSum := 0
    for _, path := range allPaths {
        num, _ := strconv.Atoi(path)
        totalSum += num
    }
    
    return totalSum
}
```

### Complexity
- **Time Complexity:** O(n) where n is the number of nodes
- **Space Complexity:** O(h) where h is the height of the tree
