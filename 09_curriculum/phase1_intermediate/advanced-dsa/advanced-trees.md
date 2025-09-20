# Advanced Trees

## Overview

This module covers advanced tree data structures including AVL trees, Red-Black trees, B-trees, Segment trees, Fenwick trees, and Tries. These structures are essential for efficient data storage and retrieval in complex applications.

## Table of Contents

1. [AVL Trees](#avl-trees)
2. [Red-Black Trees](#red-black-trees)
3. [B-Trees](#b-trees)
4. [Segment Trees](#segment-trees)
5. [Fenwick Trees (Binary Indexed Trees)](#fenwick-trees-binary-indexed-trees)
6. [Tries](#tries)
7. [Applications](#applications)
8. [Complexity Analysis](#complexity-analysis)
9. [Follow-up Questions](#follow-up-questions)

## AVL Trees

### Theory

AVL (Adelson-Velsky and Landis) trees are self-balancing binary search trees where the height difference between left and right subtrees of any node is at most 1. This ensures O(log n) time complexity for all operations.

### Key Properties

- **Balance Factor**: Height of left subtree - Height of right subtree
- **Valid Range**: -1, 0, or 1 for all nodes
- **Rotation Operations**: Left, Right, Left-Right, Right-Left

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type AVLNode struct {
    key    int
    height int
    left   *AVLNode
    right  *AVLNode
}

type AVLTree struct {
    root *AVLNode
}

func NewAVLTree() *AVLTree {
    return &AVLTree{}
}

func (n *AVLNode) getHeight() int {
    if n == nil {
        return 0
    }
    return n.height
}

func (n *AVLNode) getBalance() int {
    if n == nil {
        return 0
    }
    return n.left.getHeight() - n.right.getHeight()
}

func (n *AVLNode) updateHeight() {
    n.height = 1 + int(math.Max(float64(n.left.getHeight()), float64(n.right.getHeight())))
}

func (t *AVLTree) rightRotate(y *AVLNode) *AVLNode {
    x := y.left
    T2 := x.right

    x.right = y
    y.left = T2

    y.updateHeight()
    x.updateHeight()

    return x
}

func (t *AVLTree) leftRotate(x *AVLNode) *AVLNode {
    y := x.right
    T2 := y.left

    y.left = x
    x.right = T2

    x.updateHeight()
    y.updateHeight()

    return y
}

func (t *AVLTree) insert(node *AVLNode, key int) *AVLNode {
    if node == nil {
        return &AVLNode{key: key, height: 1}
    }

    if key < node.key {
        node.left = t.insert(node.left, key)
    } else if key > node.key {
        node.right = t.insert(node.right, key)
    } else {
        return node // Duplicate keys not allowed
    }

    node.updateHeight()

    balance := node.getBalance()

    // Left Left Case
    if balance > 1 && key < node.left.key {
        return t.rightRotate(node)
    }

    // Right Right Case
    if balance < -1 && key > node.right.key {
        return t.leftRotate(node)
    }

    // Left Right Case
    if balance > 1 && key > node.left.key {
        node.left = t.leftRotate(node.left)
        return t.rightRotate(node)
    }

    // Right Left Case
    if balance < -1 && key < node.right.key {
        node.right = t.rightRotate(node.right)
        return t.leftRotate(node)
    }

    return node
}

func (t *AVLTree) Insert(key int) {
    t.root = t.insert(t.root, key)
}

func (t *AVLTree) search(node *AVLNode, key int) *AVLNode {
    if node == nil || node.key == key {
        return node
    }

    if key < node.key {
        return t.search(node.left, key)
    }
    return t.search(node.right, key)
}

func (t *AVLTree) Search(key int) *AVLNode {
    return t.search(t.root, key)
}

func (t *AVLTree) inorder(node *AVLNode) {
    if node != nil {
        t.inorder(node.left)
        fmt.Printf("%d ", node.key)
        t.inorder(node.right)
    }
}

func (t *AVLTree) Inorder() {
    t.inorder(t.root)
    fmt.Println()
}

func main() {
    tree := NewAVLTree()
    
    keys := []int{10, 20, 30, 40, 50, 25}
    for _, key := range keys {
        tree.Insert(key)
        fmt.Printf("Inserted %d\n", key)
    }
    
    fmt.Print("Inorder traversal: ")
    tree.Inorder()
    
    // Search for a key
    result := tree.Search(30)
    if result != nil {
        fmt.Printf("Found key: %d\n", result.key)
    } else {
        fmt.Println("Key not found")
    }
}
```

#### Node.js Implementation

```javascript
class AVLNode {
    constructor(key) {
        this.key = key;
        this.height = 1;
        this.left = null;
        this.right = null;
    }
}

class AVLTree {
    constructor() {
        this.root = null;
    }

    getHeight(node) {
        return node ? node.height : 0;
    }

    getBalance(node) {
        return node ? this.getHeight(node.left) - this.getHeight(node.right) : 0;
    }

    updateHeight(node) {
        if (node) {
            node.height = 1 + Math.max(this.getHeight(node.left), this.getHeight(node.right));
        }
    }

    rightRotate(y) {
        const x = y.left;
        const T2 = x.right;

        x.right = y;
        y.left = T2;

        this.updateHeight(y);
        this.updateHeight(x);

        return x;
    }

    leftRotate(x) {
        const y = x.right;
        const T2 = y.left;

        y.left = x;
        x.right = T2;

        this.updateHeight(x);
        this.updateHeight(y);

        return y;
    }

    insert(node, key) {
        if (!node) {
            return new AVLNode(key);
        }

        if (key < node.key) {
            node.left = this.insert(node.left, key);
        } else if (key > node.key) {
            node.right = this.insert(node.right, key);
        } else {
            return node; // Duplicate keys not allowed
        }

        this.updateHeight(node);

        const balance = this.getBalance(node);

        // Left Left Case
        if (balance > 1 && key < node.left.key) {
            return this.rightRotate(node);
        }

        // Right Right Case
        if (balance < -1 && key > node.right.key) {
            return this.leftRotate(node);
        }

        // Left Right Case
        if (balance > 1 && key > node.left.key) {
            node.left = this.leftRotate(node.left);
            return this.rightRotate(node);
        }

        // Right Left Case
        if (balance < -1 && key < node.right.key) {
            node.right = this.rightRotate(node.right);
            return this.leftRotate(node);
        }

        return node;
    }

    insertKey(key) {
        this.root = this.insert(this.root, key);
    }

    search(node, key) {
        if (!node || node.key === key) {
            return node;
        }

        if (key < node.key) {
            return this.search(node.left, key);
        }
        return this.search(node.right, key);
    }

    searchKey(key) {
        return this.search(this.root, key);
    }

    inorder(node) {
        if (node) {
            this.inorder(node.left);
            process.stdout.write(`${node.key} `);
            this.inorder(node.right);
        }
    }

    printInorder() {
        this.inorder(this.root);
        console.log();
    }
}

// Example usage
const tree = new AVLTree();
const keys = [10, 20, 30, 40, 50, 25];

keys.forEach(key => {
    tree.insertKey(key);
    console.log(`Inserted ${key}`);
});

console.log('Inorder traversal:');
tree.printInorder();

// Search for a key
const result = tree.searchKey(30);
if (result) {
    console.log(`Found key: ${result.key}`);
} else {
    console.log('Key not found');
}
```

## Red-Black Trees

### Theory

Red-Black trees are self-balancing binary search trees where each node has a color (red or black) and follows specific rules to maintain balance. They provide O(log n) time complexity for all operations.

### Key Properties

1. Every node is either red or black
2. The root is always black
3. All leaves (NIL nodes) are black
4. If a node is red, both its children are black
5. All paths from root to leaves have the same number of black nodes

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
)

type Color bool

const (
    RED   Color = true
    BLACK Color = false
)

type RBNode struct {
    key    int
    color  Color
    left   *RBNode
    right  *RBNode
    parent *RBNode
}

type RBTree struct {
    root *RBNode
    nil  *RBNode
}

func NewRBTree() *RBTree {
    nilNode := &RBNode{color: BLACK}
    return &RBTree{
        root: nilNode,
        nil:  nilNode,
    }
}

func (t *RBTree) leftRotate(x *RBNode) {
    y := x.right
    x.right = y.left

    if y.left != t.nil {
        y.left.parent = x
    }

    y.parent = x.parent

    if x.parent == t.nil {
        t.root = y
    } else if x == x.parent.left {
        x.parent.left = y
    } else {
        x.parent.right = y
    }

    y.left = x
    x.parent = y
}

func (t *RBTree) rightRotate(y *RBNode) {
    x := y.left
    y.left = x.right

    if x.right != t.nil {
        x.right.parent = y
    }

    x.parent = y.parent

    if y.parent == t.nil {
        t.root = x
    } else if y == y.parent.left {
        y.parent.left = x
    } else {
        y.parent.right = x
    }

    x.right = y
    y.parent = x
}

func (t *RBTree) insertFixup(z *RBNode) {
    for z.parent.color == RED {
        if z.parent == z.parent.parent.left {
            y := z.parent.parent.right
            if y.color == RED {
                z.parent.color = BLACK
                y.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            } else {
                if z == z.parent.right {
                    z = z.parent
                    t.leftRotate(z)
                }
                z.parent.color = BLACK
                z.parent.parent.color = RED
                t.rightRotate(z.parent.parent)
            }
        } else {
            y := z.parent.parent.left
            if y.color == RED {
                z.parent.color = BLACK
                y.color = BLACK
                z.parent.parent.color = RED
                z = z.parent.parent
            } else {
                if z == z.parent.left {
                    z = z.parent
                    t.rightRotate(z)
                }
                z.parent.color = BLACK
                z.parent.parent.color = RED
                t.leftRotate(z.parent.parent)
            }
        }
    }
    t.root.color = BLACK
}

func (t *RBTree) insert(key int) {
    z := &RBNode{
        key:    key,
        color:  RED,
        left:   t.nil,
        right:  t.nil,
        parent: t.nil,
    }

    y := t.nil
    x := t.root

    for x != t.nil {
        y = x
        if z.key < x.key {
            x = x.left
        } else {
            x = x.right
        }
    }

    z.parent = y

    if y == t.nil {
        t.root = z
    } else if z.key < y.key {
        y.left = z
    } else {
        y.right = z
    }

    t.insertFixup(z)
}

func (t *RBTree) Insert(key int) {
    t.insert(key)
}

func (t *RBTree) search(node *RBNode, key int) *RBNode {
    if node == t.nil || node.key == key {
        return node
    }

    if key < node.key {
        return t.search(node.left, key)
    }
    return t.search(node.right, key)
}

func (t *RBTree) Search(key int) *RBNode {
    return t.search(t.root, key)
}

func (t *RBTree) inorder(node *RBNode) {
    if node != t.nil {
        t.inorder(node.left)
        color := "BLACK"
        if node.color == RED {
            color = "RED"
        }
        fmt.Printf("%d(%s) ", node.key, color)
        t.inorder(node.right)
    }
}

func (t *RBTree) Inorder() {
    t.inorder(t.root)
    fmt.Println()
}

func main() {
    tree := NewRBTree()
    
    keys := []int{10, 20, 30, 40, 50, 25}
    for _, key := range keys {
        tree.Insert(key)
        fmt.Printf("Inserted %d\n", key)
    }
    
    fmt.Print("Inorder traversal: ")
    tree.Inorder()
    
    // Search for a key
    result := tree.Search(30)
    if result != tree.nil {
        fmt.Printf("Found key: %d\n", result.key)
    } else {
        fmt.Println("Key not found")
    }
}
```

## B-Trees

### Theory

B-trees are self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in O(log n) time. They are commonly used in databases and file systems.

### Key Properties

- All leaves are at the same level
- Every node except root has at least t-1 keys and at most 2t-1 keys
- Root has at least 1 key
- Internal nodes have at least t children and at most 2t children

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
)

type BTreeNode struct {
    keys     []int
    children []*BTreeNode
    leaf     bool
    n        int // number of keys
}

type BTree struct {
    root *BTreeNode
    t    int // minimum degree
}

func NewBTree(t int) *BTree {
    return &BTree{t: t}
}

func (bt *BTree) createNode() *BTreeNode {
    return &BTreeNode{
        keys:     make([]int, 2*bt.t-1),
        children: make([]*BTreeNode, 2*bt.t),
        leaf:     true,
        n:        0,
    }
}

func (bt *BTree) splitChild(x *BTreeNode, i int) {
    t := bt.t
    y := x.children[i]
    z := bt.createNode()
    z.leaf = y.leaf
    z.n = t - 1

    // Copy the last t-1 keys of y to z
    for j := 0; j < t-1; j++ {
        z.keys[j] = y.keys[j+t]
    }

    // Copy the last t children of y to z
    if !y.leaf {
        for j := 0; j < t; j++ {
            z.children[j] = y.children[j+t]
        }
    }

    y.n = t - 1

    // Make space for new child
    for j := x.n; j >= i+1; j-- {
        x.children[j+1] = x.children[j]
    }

    x.children[i+1] = z

    // Move median key to x
    for j := x.n - 1; j >= i; j-- {
        x.keys[j+1] = x.keys[j]
    }

    x.keys[i] = y.keys[t-1]
    x.n++
}

func (bt *BTree) insertNonFull(x *BTreeNode, k int) {
    i := x.n - 1

    if x.leaf {
        // Find position for new key
        for i >= 0 && k < x.keys[i] {
            x.keys[i+1] = x.keys[i]
            i--
        }
        x.keys[i+1] = k
        x.n++
    } else {
        // Find child to insert into
        for i >= 0 && k < x.keys[i] {
            i--
        }
        i++

        if x.children[i].n == 2*bt.t-1 {
            bt.splitChild(x, i)
            if k > x.keys[i] {
                i++
            }
        }
        bt.insertNonFull(x.children[i], k)
    }
}

func (bt *BTree) Insert(k int) {
    if bt.root == nil {
        bt.root = bt.createNode()
        bt.root.keys[0] = k
        bt.root.n = 1
        return
    }

    if bt.root.n == 2*bt.t-1 {
        s := bt.createNode()
        s.leaf = false
        s.children[0] = bt.root
        bt.root = s
        bt.splitChild(s, 0)
    }

    bt.insertNonFull(bt.root, k)
}

func (bt *BTree) search(x *BTreeNode, k int) *BTreeNode {
    i := 0
    for i < x.n && k > x.keys[i] {
        i++
    }

    if i < x.n && k == x.keys[i] {
        return x
    }

    if x.leaf {
        return nil
    }

    return bt.search(x.children[i], k)
}

func (bt *BTree) Search(k int) *BTreeNode {
    return bt.search(bt.root, k)
}

func (bt *BTree) traverse(x *BTreeNode) {
    i := 0
    for i < x.n {
        if !x.leaf {
            bt.traverse(x.children[i])
        }
        fmt.Printf("%d ", x.keys[i])
        i++
    }
    if !x.leaf {
        bt.traverse(x.children[i])
    }
}

func (bt *BTree) Traverse() {
    if bt.root != nil {
        bt.traverse(bt.root)
        fmt.Println()
    }
}

func main() {
    tree := NewBTree(3) // Minimum degree 3
    
    keys := []int{10, 20, 5, 6, 12, 30, 7, 17}
    for _, key := range keys {
        tree.Insert(key)
        fmt.Printf("Inserted %d\n", key)
    }
    
    fmt.Print("B-Tree traversal: ")
    tree.Traverse()
    
    // Search for a key
    result := tree.Search(6)
    if result != nil {
        fmt.Printf("Found key: 6\n")
    } else {
        fmt.Println("Key not found")
    }
}
```

## Segment Trees

### Theory

Segment trees are data structures that allow efficient range queries and updates on an array. They support operations like range sum, range minimum, range maximum, etc., in O(log n) time.

### Key Properties

- Leaf nodes represent array elements
- Internal nodes represent ranges
- Each node stores information about its range
- Supports range queries and point updates

### Implementations

#### Golang Implementation

```go
package main

import (
    "fmt"
    "math"
)

type SegmentTree struct {
    tree []int
    n    int
}

func NewSegmentTree(arr []int) *SegmentTree {
    n := len(arr)
    size := 1
    for size < n {
        size *= 2
    }
    
    st := &SegmentTree{
        tree: make([]int, 2*size),
        n:    n,
    }
    
    // Initialize leaves
    for i := 0; i < n; i++ {
        st.tree[size+i] = arr[i]
    }
    
    // Build the tree
    for i := size - 1; i > 0; i-- {
        st.tree[i] = st.tree[2*i] + st.tree[2*i+1]
    }
    
    return st
}

func (st *SegmentTree) update(pos int, value int) {
    pos += st.n
    st.tree[pos] = value
    
    for pos > 1 {
        pos /= 2
        st.tree[pos] = st.tree[2*pos] + st.tree[2*pos+1]
    }
}

func (st *SegmentTree) query(l, r int) int {
    l += st.n
    r += st.n
    sum := 0
    
    for l <= r {
        if l%2 == 1 {
            sum += st.tree[l]
            l++
        }
        if r%2 == 0 {
            sum += st.tree[r]
            r--
        }
        l /= 2
        r /= 2
    }
    
    return sum
}

func main() {
    arr := []int{1, 3, 5, 7, 9, 11}
    st := NewSegmentTree(arr)
    
    fmt.Printf("Array: %v\n", arr)
    fmt.Printf("Sum of range [1, 3]: %d\n", st.query(1, 3))
    fmt.Printf("Sum of range [0, 5]: %d\n", st.query(0, 5))
    
    // Update element at index 1 to 10
    st.update(1, 10)
    fmt.Printf("After updating index 1 to 10:\n")
    fmt.Printf("Sum of range [1, 3]: %d\n", st.query(1, 3))
}
```

## Follow-up Questions

### 1. Tree Selection
**Q: When would you choose AVL trees over Red-Black trees?**
A: AVL trees provide better balance (height difference ≤ 1) leading to faster lookups, but more rotations during insertions. Use AVL when you have more lookups than insertions. Red-Black trees have fewer rotations but slightly less balanced, making them better for frequent insertions.

### 2. B-Tree Applications
**Q: Why are B-trees commonly used in databases?**
A: B-trees minimize disk I/O by keeping more data in each node, reducing the number of disk accesses. They maintain sorted order and support efficient range queries, making them ideal for database indexes.

### 3. Segment Tree Optimization
**Q: How can you optimize segment trees for different operations?**
A: Use lazy propagation for range updates, implement different aggregation functions (min, max, sum), and consider using sparse segment trees for large ranges with few updates.

## Complexity Analysis

| Operation | AVL Tree | Red-Black Tree | B-Tree | Segment Tree |
|-----------|----------|----------------|--------|--------------|
| Search    | O(log n) | O(log n)       | O(log n) | O(log n) |
| Insert    | O(log n) | O(log n)       | O(log n) | O(log n) |
| Delete    | O(log n) | O(log n)       | O(log n) | O(log n) |
| Space     | O(n)     | O(n)           | O(n)    | O(n) |

## Applications

1. **AVL Trees**: Database indexes, memory management
2. **Red-Black Trees**: C++ STL map/set, Java TreeMap
3. **B-Trees**: Database storage, file systems
4. **Segment Trees**: Range queries, competitive programming
5. **Tries**: Autocomplete, spell checkers, IP routing

---

**Next**: [Graph Algorithms](graph-algorithms.md/) | **Previous**: [Advanced DSA](README.md/) | **Up**: [Advanced DSA](README.md/)
