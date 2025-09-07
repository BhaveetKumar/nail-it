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
1. **Greedy Choice Property**: Make the locally optimal choice at each step
2. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
3. **No Backtracking**: Once a choice is made, it's never reconsidered
4. **Local Optimization**: Choose the best option available at each step

### **When to Use Greedy**
- **Optimization Problems**: Find minimum/maximum value
- **Scheduling Problems**: Optimize resource allocation
- **Selection Problems**: Choose optimal subset
- **MST Problems**: Minimum spanning tree algorithms

### **Common Greedy Patterns**
- **Sorting First**: Sort by some criteria, then apply greedy choice
- **Two Pointers**: Use pointers to make optimal choices
- **Priority Queue**: Use heap for dynamic optimal choices
- **Interval Scheduling**: Sort by end time, select non-overlapping intervals

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
