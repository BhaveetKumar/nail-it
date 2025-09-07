# Queue Reconstruction by Height

### Problem
You are given an array of people, `people`, which are the attributes of some people in a queue (not necessarily in order). Each `people[i] = [hi, ki]` represents the `ith` person of height `hi` with exactly `ki` other people in front who have a height greater than or equal to `hi`.

Reconstruct and return the queue that is represented by the input array `people`. The returned queue should be formatted as an array `queue`, where `queue[j] = [hj, kj]` is the attributes of the `jth` person in the queue (`queue[0]` is the person at the front of the queue).

**Example:**
```
Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
Explanation:
Person 0 has height 5 with no other people taller or the same height in front.
Person 1 has height 7 with no other people taller or the same height in front.
Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
Person 3 has height 6 with one person taller or the same height in front, which is person 1.
Person 4 has height 4 with four people taller or the same height in front, which are persons 0, 1, 2, and 3.
Person 5 has height 7 with one person taller or the same height in front, which is person 1.
Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
```

### Golang Solution

```go
import "sort"

func reconstructQueue(people [][]int) [][]int {
    // Sort by height (descending) and then by k (ascending)
    sort.Slice(people, func(i, j int) bool {
        if people[i][0] == people[j][0] {
            return people[i][1] < people[j][1]
        }
        return people[i][0] > people[j][0]
    })
    
    var result [][]int
    
    for _, person := range people {
        // Insert at position k
        k := person[1]
        result = append(result[:k], append([][]int{person}, result[k:]...)...)
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Linked List**
```go
type ListNode struct {
    Val  []int
    Next *ListNode
}

func reconstructQueueLinkedList(people [][]int) [][]int {
    // Sort by height (descending) and then by k (ascending)
    sort.Slice(people, func(i, j int) bool {
        if people[i][0] == people[j][0] {
            return people[i][1] < people[j][1]
        }
        return people[i][0] > people[j][0]
    })
    
    var head *ListNode
    
    for _, person := range people {
        k := person[1]
        newNode := &ListNode{Val: person}
        
        if head == nil {
            head = newNode
        } else {
            current := head
            var prev *ListNode
            
            for i := 0; i < k; i++ {
                prev = current
                current = current.Next
            }
            
            if prev == nil {
                newNode.Next = head
                head = newNode
            } else {
                newNode.Next = current
                prev.Next = newNode
            }
        }
    }
    
    // Convert to array
    var result [][]int
    current := head
    for current != nil {
        result = append(result, current.Val)
        current = current.Next
    }
    
    return result
}
```

#### **Using Binary Search Tree**
```go
type BSTNode struct {
    Val   []int
    Left  *BSTNode
    Right *BSTNode
    Size  int
}

func reconstructQueueBST(people [][]int) [][]int {
    // Sort by height (descending) and then by k (ascending)
    sort.Slice(people, func(i, j int) bool {
        if people[i][0] == people[j][0] {
            return people[i][1] < people[j][1]
        }
        return people[i][0] > people[j][0]
    })
    
    var root *BSTNode
    
    for _, person := range people {
        k := person[1]
        root = insertBST(root, person, k)
    }
    
    // In-order traversal to get result
    var result [][]int
    inorderBST(root, &result)
    
    return result
}

func insertBST(root *BSTNode, person []int, k int) *BSTNode {
    if root == nil {
        return &BSTNode{Val: person, Size: 1}
    }
    
    if k <= root.Size {
        root.Left = insertBST(root.Left, person, k)
        root.Size++
    } else {
        root.Right = insertBST(root.Right, person, k-root.Size-1)
        root.Size++
    }
    
    return root
}

func inorderBST(root *BSTNode, result *[][]int) {
    if root == nil {
        return
    }
    
    inorderBST(root.Left, result)
    *result = append(*result, root.Val)
    inorderBST(root.Right, result)
}
```

#### **Brute Force**
```go
func reconstructQueueBruteForce(people [][]int) [][]int {
    n := len(people)
    used := make([]bool, n)
    result := make([][]int, n)
    
    var backtrack func(int)
    backtrack = func(pos int) bool {
        if pos == n {
            return true
        }
        
        for i := 0; i < n; i++ {
            if used[i] {
                continue
            }
            
            person := people[i]
            count := 0
            
            // Count taller or equal height people in current result
            for j := 0; j < pos; j++ {
                if result[j][0] >= person[0] {
                    count++
                }
            }
            
            if count == person[1] {
                result[pos] = person
                used[i] = true
                
                if backtrack(pos + 1) {
                    return true
                }
                
                used[i] = false
            }
        }
        
        return false
    }
    
    backtrack(0)
    return result
}
```

#### **Return with Validation**
```go
func reconstructQueueWithValidation(people [][]int) ([][]int, bool) {
    result := reconstructQueue(people)
    
    // Validate the result
    for i := 0; i < len(result); i++ {
        person := result[i]
        count := 0
        
        for j := 0; j < i; j++ {
            if result[j][0] >= person[0] {
                count++
            }
        }
        
        if count != person[1] {
            return result, false
        }
    }
    
    return result, true
}
```

### Complexity
- **Time Complexity:** O(n²) for insertion, O(n log n) for sorting
- **Space Complexity:** O(n)
