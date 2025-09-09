# Greedy Pattern

> **Master greedy algorithms and optimization strategies with Go implementations**

## 📋 Problems

### **Activity Selection**

- [Jump Game](./JumpGame.md) - Can reach the last index
- [Jump Game II](./JumpGameII.md) - Minimum jumps to reach end
- [Gas Station](./GasStation.md) - Find starting gas station
- [Container With Most Water](./ContainerWithMostWater.md) - Maximum area between lines
- [Trapping Rain Water](./TrappingRainWater.md) - Collect rainwater

### **Scheduling Problems**

- [Meeting Rooms](./MeetingRooms.md) - Can attend all meetings
- [Meeting Rooms II](./MeetingRoomsII.md) - Minimum meeting rooms needed
- [Task Scheduler](./TaskScheduler.md) - Schedule tasks with cooldown
- [Course Schedule III](./CourseScheduleIII.md) - Maximum courses to take
- [Minimum Number of Arrows](./MinimumNumberOfArrows.md) - Burst balloons

### **Optimization Problems**

- [Maximum Subarray](./MaximumSubarray.md) - Kadane's algorithm
- [Best Time to Buy and Sell Stock](./BestTimeToBuyAndSellStock.md) - Maximum profit
- [Best Time to Buy and Sell Stock II](./BestTimeToBuyAndSellStockII.md) - Multiple transactions
- [Lemonade Change](./LemonadeChange.md) - Make change with bills
- [Assign Cookies](./AssignCookies.md) - Distribute cookies to children

### **Advanced Greedy**

- [Huffman Coding](./HuffmanCoding.md) - Optimal prefix coding
- [Fractional Knapsack](./FractionalKnapsack.md) - Greedy knapsack
- [Minimum Cost to Connect Sticks](./MinimumCostToConnectSticks.md) - Connect sticks optimally
- [Reorganize String](./ReorganizeString.md) - Rearrange string characters
- [Partition Labels](./PartitionLabels.md) - Partition string into labels

---

## 🎯 Key Concepts

### **Greedy Algorithm Principles**

**Detailed Explanation:**
Greedy algorithms are a class of algorithms that make locally optimal choices at each step with the hope of finding a global optimum. They are characterized by their simplicity and efficiency, but they don't always guarantee the optimal solution for all problems.

**Core Principles:**

**1. Greedy Choice Property:**

- **Definition**: A globally optimal solution can be arrived at by making a locally optimal choice
- **Key Insight**: The choice made at each step should be the best choice available at that moment
- **Example**: In activity selection, always choose the activity with the earliest finish time
- **Mathematical Foundation**: If a greedy choice is optimal, then the remaining subproblem is also optimal

**2. Optimal Substructure:**

- **Definition**: An optimal solution to the problem contains optimal solutions to its subproblems
- **Implication**: The problem can be broken down into smaller, similar subproblems
- **Example**: In minimum spanning tree, the optimal tree for the entire graph contains optimal trees for subgraphs
- **Relationship**: Works in conjunction with greedy choice property

**3. No Backtracking:**

- **Definition**: Once a choice is made, it's never reconsidered or undone
- **Efficiency**: This makes greedy algorithms very efficient in terms of time and space
- **Trade-off**: May lead to suboptimal solutions if the greedy choice property doesn't hold
- **Example**: Once an activity is selected in activity selection, it's never removed

**4. Local Optimization:**

- **Definition**: At each step, choose the option that seems best at that moment
- **Strategy**: Don't consider future consequences of current choices
- **Benefit**: Simple to implement and understand
- **Limitation**: May not lead to global optimum

**Why Greedy Algorithms Work:**

- **Mathematical Proof**: Many greedy algorithms can be proven to be optimal
- **Intuitive Approach**: Often mirror human decision-making processes
- **Efficiency**: Typically have O(n log n) or O(n) time complexity
- **Simplicity**: Easier to implement than dynamic programming solutions

### **When to Use Greedy**

**Detailed Explanation:**
Greedy algorithms are most effective when the problem exhibits certain characteristics that make local optimization lead to global optimization.

**Problem Characteristics:**

**1. Optimization Problems:**

- **Nature**: Problems where you need to find minimum/maximum value
- **Examples**: Minimum cost, maximum profit, shortest path
- **Strategy**: Make choices that optimize the current step
- **Validation**: Ensure greedy choice leads to optimal solution

**2. Scheduling Problems:**

- **Nature**: Problems involving resource allocation and time management
- **Examples**: Meeting room scheduling, task scheduling, job sequencing
- **Strategy**: Sort by relevant criteria and make greedy choices
- **Common Pattern**: Sort by end time, start time, or priority

**3. Selection Problems:**

- **Nature**: Problems where you need to choose an optimal subset
- **Examples**: Activity selection, knapsack (fractional), interval scheduling
- **Strategy**: Use greedy criteria to select elements
- **Validation**: Prove that greedy selection leads to optimal solution

**4. MST Problems:**

- **Nature**: Problems involving minimum spanning trees
- **Examples**: Kruskal's algorithm, Prim's algorithm
- **Strategy**: Always add the minimum weight edge that doesn't create a cycle
- **Proof**: Can be proven to be optimal

**When NOT to Use Greedy:**

- **Problems without optimal substructure**: Can't break into smaller subproblems
- **Problems where greedy choice doesn't work**: Local optimum doesn't lead to global optimum
- **Problems requiring backtracking**: Need to reconsider previous choices
- **Complex optimization**: Where dynamic programming is more appropriate

### **Common Greedy Patterns**

**Detailed Explanation:**
Greedy algorithms often follow specific patterns that can be recognized and applied to solve similar problems.

**1. Sorting First:**

- **Pattern**: Sort the input by some criteria, then apply greedy choice
- **Why it works**: Sorting helps identify the optimal order for making choices
- **Examples**: Activity selection (sort by end time), fractional knapsack (sort by value/weight ratio)
- **Time Complexity**: O(n log n) for sorting + O(n) for greedy = O(n log n)
- **Go Implementation**: Use `sort.Slice()` with custom comparison functions

**2. Two Pointers:**

- **Pattern**: Use two pointers to make optimal choices from opposite ends
- **Why it works**: Often the optimal choice involves elements from both ends
- **Examples**: Container with most water, two sum in sorted array
- **Time Complexity**: O(n) - single pass through the array
- **Go Implementation**: Use left and right pointers with conditional movement

**3. Priority Queue (Heap):**

- **Pattern**: Use heap to dynamically find the optimal choice
- **Why it works**: Heap maintains the optimal element at the top
- **Examples**: Minimum cost to connect sticks, merge k sorted lists
- **Time Complexity**: O(n log n) for heap operations
- **Go Implementation**: Use `container/heap` package with custom heap interface

**4. Interval Scheduling:**

- **Pattern**: Sort intervals by end time, select non-overlapping intervals
- **Why it works**: Selecting earliest finishing interval leaves maximum room for other intervals
- **Examples**: Activity selection, meeting room scheduling
- **Time Complexity**: O(n log n) for sorting + O(n) for selection
- **Go Implementation**: Sort intervals and iterate with greedy selection

**Advanced Patterns:**

- **Exchange Argument**: Prove optimality by showing any other solution can be transformed to greedy solution
- **Cut-and-Paste**: Show that replacing part of optimal solution with greedy choice doesn't worsen it
- **Induction**: Use mathematical induction to prove greedy algorithm correctness

**Discussion Questions & Answers:**

**Q1: How do you prove that a greedy algorithm produces the optimal solution?**

**Answer:** Proving greedy algorithm optimality:

- **Greedy Choice Property**: Show that a globally optimal solution can be arrived at by making a locally optimal choice
- **Optimal Substructure**: Prove that the problem has optimal substructure property
- **Exchange Argument**: Show that any optimal solution can be transformed to match the greedy solution
- **Cut-and-Paste Method**: Demonstrate that replacing part of optimal solution with greedy choice doesn't worsen it
- **Induction**: Use mathematical induction to prove correctness
- **Counterexample**: Show that no other choice can lead to a better solution
- **Mathematical Proof**: Provide formal mathematical proof of optimality
- **Complexity Analysis**: Ensure the algorithm runs in optimal time complexity

**Q2: What are the common pitfalls when implementing greedy algorithms in Go?**

**Answer:** Common pitfalls include:

- **Incorrect Sorting**: Sorting by wrong criteria or using incorrect comparison function
- **Edge Cases**: Not handling empty arrays, single elements, or boundary conditions
- **Type Conversions**: Issues with type conversions in custom sorting functions
- **Slice Bounds**: Index out of bounds errors when accessing sorted arrays
- **Heap Implementation**: Incorrect heap interface implementation or heap property violations
- **Pointer Management**: Issues with two-pointer technique and boundary conditions
- **Overflow**: Integer overflow in calculations, especially with large numbers
- **Validation**: Not validating that greedy choice actually leads to optimal solution

**Q3: How do you optimize greedy algorithms for performance in Go?**

**Answer:** Performance optimization strategies:

- **Efficient Sorting**: Use `sort.Slice()` for custom sorting instead of `sort.Sort()`
- **Heap Optimization**: Implement heap interface correctly and use `container/heap` package
- **Memory Management**: Reuse slices and avoid unnecessary allocations
- **Early Termination**: Stop processing when optimal solution is found
- **In-place Operations**: Modify arrays in-place when possible to save memory
- **Parallel Processing**: Use goroutines for independent greedy operations when applicable
- **Profiling**: Use `go tool pprof` to identify performance bottlenecks
- **Algorithm Selection**: Choose the most efficient greedy approach for the specific problem

---

## 🛠️ Go-Specific Tips

### **Sorting for Greedy**

```go
import "sort"

// Sort by custom criteria
type Interval struct {
    Start int
    End   int
}

func canAttendMeetings(intervals []Interval) bool {
    // Sort by start time
    sort.Slice(intervals, func(i, j int) bool {
        return intervals[i].Start < intervals[j].Start
    })

    // Check for overlaps
    for i := 1; i < len(intervals); i++ {
        if intervals[i].Start < intervals[i-1].End {
            return false
        }
    }

    return true
}
```

### **Priority Queue (Heap)**

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

func minCostConnectSticks(sticks []int) int {
    h := &IntHeap{}
    heap.Init(h)

    for _, stick := range sticks {
        heap.Push(h, stick)
    }

    cost := 0
    for h.Len() > 1 {
        first := heap.Pop(h).(int)
        second := heap.Pop(h).(int)
        combined := first + second
        cost += combined
        heap.Push(h, combined)
    }

    return cost
}
```

### **Two Pointers Technique**

```go
func maxArea(height []int) int {
    left, right := 0, len(height)-1
    maxArea := 0

    for left < right {
        width := right - left
        currentArea := min(height[left], height[right]) * width
        maxArea = max(maxArea, currentArea)

        // Greedy choice: move the pointer with smaller height
        if height[left] < height[right] {
            left++
        } else {
            right--
        }
    }

    return maxArea
}
```

### **Greedy with Validation**

```go
func canJump(nums []int) bool {
    maxReach := 0

    for i := 0; i < len(nums); i++ {
        // If current position is beyond max reach, can't proceed
        if i > maxReach {
            return false
        }

        // Update max reach with greedy choice
        maxReach = max(maxReach, i+nums[i])

        // Early termination if we can reach the end
        if maxReach >= len(nums)-1 {
            return true
        }
    }

    return true
}
```

---

## 🎯 Interview Tips

### **How to Identify Greedy Problems**

1. **Optimization**: Find minimum/maximum value
2. **Local Choice**: Best choice at each step leads to global optimum
3. **No Backtracking**: Once choice is made, never reconsider
4. **Sorting**: Often involves sorting by some criteria

### **Common Greedy Problem Patterns**

- **Activity Selection**: Choose non-overlapping activities
- **Scheduling**: Optimize resource allocation
- **Knapsack**: Fractional knapsack problems
- **Graph**: Minimum spanning tree, shortest path
- **String**: Character frequency and arrangement

### **Optimization Tips**

- **Sort First**: Sort by relevant criteria
- **Use Heaps**: For dynamic optimal choices
- **Two Pointers**: For array optimization
- **Early Termination**: Stop when optimal solution is found
