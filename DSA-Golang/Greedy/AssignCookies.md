# Assign Cookies

### Problem
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child `i` has a greed factor `g[i]`, which is the minimum size of a cookie that the child will be content with; and each cookie `j` has a size `s[j]`. If `s[j] >= g[i]`, we can assign the cookie `j` to the child `i`, and the child `i` will be content. Your goal is to maximize the number of your content children and output the maximum number.

**Example:**
```
Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.

Input: g = [1,2], s = [1,2,3]
Output: 2
Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
You have 3 cookies and their sizes are big enough to gratify all of the children, 
You need to output 2.
```

### Golang Solution

```go
import "sort"

func findContentChildren(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    
    childIndex := 0
    cookieIndex := 0
    contentChildren := 0
    
    for childIndex < len(g) && cookieIndex < len(s) {
        if s[cookieIndex] >= g[childIndex] {
            contentChildren++
            childIndex++
        }
        cookieIndex++
    }
    
    return contentChildren
}
```

### Alternative Solutions

#### **Using Two Pointers**
```go
func findContentChildrenTwoPointers(g []int, s []int) int {
    sort.Ints(g)
    sort.Ints(s)
    
    i, j := 0, 0
    count := 0
    
    for i < len(g) && j < len(s) {
        if g[i] <= s[j] {
            count++
            i++
        }
        j++
    }
    
    return count
}
```

#### **Greedy with Priority Queue**
```go
import "container/heap"

type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
    *h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[0 : n-1]
    return x
}

func findContentChildrenPQ(g []int, s []int) int {
    // Create min heaps for children and cookies
    childHeap := &IntHeap{}
    cookieHeap := &IntHeap{}
    
    heap.Init(childHeap)
    heap.Init(cookieHeap)
    
    for _, greed := range g {
        heap.Push(childHeap, greed)
    }
    
    for _, size := range s {
        heap.Push(cookieHeap, size)
    }
    
    count := 0
    
    for childHeap.Len() > 0 && cookieHeap.Len() > 0 {
        childGreed := (*childHeap)[0]
        cookieSize := (*cookieHeap)[0]
        
        if cookieSize >= childGreed {
            heap.Pop(childHeap)
            heap.Pop(cookieHeap)
            count++
        } else {
            heap.Pop(cookieHeap)
        }
    }
    
    return count
}
```

#### **Brute Force**
```go
func findContentChildrenBruteForce(g []int, s []int) int {
    used := make([]bool, len(s))
    count := 0
    
    for _, greed := range g {
        bestCookie := -1
        bestSize := math.MaxInt32
        
        for i, size := range s {
            if !used[i] && size >= greed && size < bestSize {
                bestCookie = i
                bestSize = size
            }
        }
        
        if bestCookie != -1 {
            used[bestCookie] = true
            count++
        }
    }
    
    return count
}
```

### Complexity
- **Time Complexity:** O(n log n + m log m) for sorting, O(n + m) for two pointers
- **Space Complexity:** O(1) for sorting, O(n + m) for priority queue
