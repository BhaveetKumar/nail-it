# Jump Game

### Problem
You are given an integer array `nums`. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return `true` if you can reach the last index, or `false` otherwise.

**Example:**
```
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
```

**Constraints:**
- 1 ≤ nums.length ≤ 10⁴
- 0 ≤ nums[i] ≤ 10⁵

### Explanation

#### **Greedy Approach**
- Keep track of the maximum reachable index
- At each position, update the maximum reach if possible
- If current position exceeds maximum reach, return false
- If maximum reach reaches or exceeds last index, return true
- Time Complexity: O(n)
- Space Complexity: O(1)

#### **Dynamic Programming Approach**
- Use DP to track if each position is reachable
- For each position, check if it can be reached from previous positions
- Time Complexity: O(n²)
- Space Complexity: O(n)

### Dry Run

**Input:** `nums = [2,3,1,1,4]`

| Step | Index | Value | MaxReach | Can Reach | Action |
|------|-------|-------|----------|-----------|---------|
| 0 | 0 | 2 | 0 | Yes | Update maxReach = 0 + 2 = 2 |
| 1 | 1 | 3 | 2 | Yes | Update maxReach = max(2, 1+3) = 4 |
| 2 | 2 | 1 | 4 | Yes | Update maxReach = max(4, 2+1) = 4 |
| 3 | 3 | 1 | 4 | Yes | Update maxReach = max(4, 3+1) = 4 |
| 4 | 4 | 4 | 4 | Yes | Reached last index! |

**Result:** `true`

**Input:** `nums = [3,2,1,0,4]`

| Step | Index | Value | MaxReach | Can Reach | Action |
|------|-------|-------|----------|-----------|---------|
| 0 | 0 | 3 | 0 | Yes | Update maxReach = 0 + 3 = 3 |
| 1 | 1 | 2 | 3 | Yes | Update maxReach = max(3, 1+2) = 3 |
| 2 | 2 | 1 | 3 | Yes | Update maxReach = max(3, 2+1) = 3 |
| 3 | 3 | 0 | 3 | Yes | Update maxReach = max(3, 3+0) = 3 |
| 4 | 4 | 4 | 3 | **No** | Can't reach index 4! |

**Result:** `false`

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

#### **Greedy Solution**
```go
func canJump(nums []int) bool {
    if len(nums) <= 1 {
        return true
    }
    
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

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

#### **Dynamic Programming Solution**
```go
func canJumpDP(nums []int) bool {
    if len(nums) <= 1 {
        return true
    }
    
    // dp[i] represents if position i is reachable
    dp := make([]bool, len(nums))
    dp[0] = true
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            // If position j is reachable and can reach position i
            if dp[j] && j+nums[j] >= i {
                dp[i] = true
                break
            }
        }
    }
    
    return dp[len(nums)-1]
}
```

### Alternative Solutions

#### **Optimized Greedy with Early Termination**
```go
func canJumpOptimized(nums []int) bool {
    if len(nums) <= 1 {
        return true
    }
    
    maxReach := 0
    target := len(nums) - 1
    
    for i := 0; i <= maxReach; i++ {
        // Update max reach
        maxReach = max(maxReach, i+nums[i])
        
        // Early termination
        if maxReach >= target {
            return true
        }
    }
    
    return false
}
```

#### **Backward Greedy Approach**
```go
func canJumpBackward(nums []int) bool {
    if len(nums) <= 1 {
        return true
    }
    
    lastPos := len(nums) - 1
    
    // Work backwards from the last position
    for i := len(nums) - 2; i >= 0; i-- {
        // If we can reach the last position from current position
        if i+nums[i] >= lastPos {
            lastPos = i
        }
    }
    
    return lastPos == 0
}
```

#### **Jump Game II - Minimum Jumps**
```go
func jump(nums []int) int {
    if len(nums) <= 1 {
        return 0
    }
    
    jumps := 0
    currentEnd := 0
    farthest := 0
    
    for i := 0; i < len(nums)-1; i++ {
        // Update the farthest we can reach
        farthest = max(farthest, i+nums[i])
        
        // If we've reached the end of current jump
        if i == currentEnd {
            jumps++
            currentEnd = farthest
            
            // If we can reach the end, no need to continue
            if currentEnd >= len(nums)-1 {
                break
            }
        }
    }
    
    return jumps
}
```

### Notes / Variations

#### **Related Problems**
- **Jump Game II**: Find minimum number of jumps
- **Jump Game III**: Jump to specific target value
- **Jump Game IV**: Jump with additional constraints
- **Jump Game V**: Jump with maximum distance limit
- **Jump Game VI**: Jump with score optimization

#### **ICPC Insights**
- **Greedy Choice**: Always choose the maximum reach at each step
- **Early Termination**: Stop as soon as we can reach the end
- **Space Optimization**: Use O(1) space instead of O(n) DP
- **Edge Cases**: Handle single element and empty arrays

#### **Go-Specific Optimizations**
```go
// Use math.Max for cleaner code (Go 1.21+)
import "math"

func canJump(nums []int) bool {
    maxReach := 0
    
    for i := 0; i < len(nums); i++ {
        if i > maxReach {
            return false
        }
        maxReach = int(math.Max(float64(maxReach), float64(i+nums[i])))
        if maxReach >= len(nums)-1 {
            return true
        }
    }
    
    return true
}

// Use pointer to avoid array bounds checking
func canJump(nums []int) bool {
    if len(nums) <= 1 {
        return true
    }
    
    maxReach := 0
    target := len(nums) - 1
    
    for i := 0; i <= maxReach && maxReach < target; i++ {
        maxReach = max(maxReach, i+nums[i])
    }
    
    return maxReach >= target
}
```

#### **Real-World Applications**
- **Game Development**: Character movement in platformers
- **Network Routing**: Find optimal paths in networks
- **Resource Allocation**: Optimize resource distribution
- **Robotics**: Path planning for robots

### Testing

```go
func TestCanJump(t *testing.T) {
    tests := []struct {
        nums     []int
        expected bool
    }{
        {[]int{2, 3, 1, 1, 4}, true},
        {[]int{3, 2, 1, 0, 4}, false},
        {[]int{0}, true},
        {[]int{1}, true},
        {[]int{0, 1}, false},
        {[]int{1, 0, 1, 0}, false},
        {[]int{2, 0, 0}, true},
        {[]int{1, 2, 3}, true},
    }
    
    for _, test := range tests {
        result := canJump(test.nums)
        if result != test.expected {
            t.Errorf("canJump(%v) = %v, expected %v", 
                test.nums, result, test.expected)
        }
    }
}
```

### Visualization

```
Input: [2, 3, 1, 1, 4]

Index: 0  1  2  3  4
Value: 2  3  1  1  4
       ↑
       Start

Step 1: From index 0, can reach max index 2
       0 -> 1 -> 2
       ↑    ↑    ↑
       Start 1   2

Step 2: From index 1, can reach max index 4
       0 -> 1 -> 2 -> 3 -> 4
       ↑    ↑    ↑    ↑    ↑
       Start 1   2   3   4 (Target reached!)

Result: true
```

### Performance Comparison

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| Greedy | O(n) | O(1) | Space efficient | Requires proof |
| DP | O(n²) | O(n) | Easy to understand | Less efficient |
| Backward Greedy | O(n) | O(1) | Alternative approach | Same complexity |

**Recommendation**: Use greedy approach for optimal performance, DP for understanding the problem structure.
