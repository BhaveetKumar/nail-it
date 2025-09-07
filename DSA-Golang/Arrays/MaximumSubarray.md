# Maximum Subarray (Kadane's Algorithm)

### Problem
Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

**Example:**
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6
```

**Constraints:**
- 1 ≤ nums.length ≤ 10⁵
- -10⁴ ≤ nums[i] ≤ 10⁴

### Explanation

#### **Brute Force Approach**
- Check all possible subarrays
- Calculate sum for each subarray and keep track of maximum
- Time Complexity: O(n³) or O(n²) with prefix sum
- Space Complexity: O(1)

#### **Kadane's Algorithm (Dynamic Programming)**
- At each position, decide whether to start a new subarray or extend the existing one
- If current element is greater than the sum so far, start a new subarray
- Otherwise, extend the existing subarray
- Time Complexity: O(n)
- Space Complexity: O(1)

### Dry Run

**Input:** `nums = [-2,1,-3,4,-1,2,1,-5,4]`

| i | nums[i] | currentSum | maxSum | Decision |
|---|---------|------------|--------|----------|
| 0 | -2 | -2 | -2 | Start new subarray |
| 1 | 1 | max(1, -2+1) = 1 | 1 | Start new subarray |
| 2 | -3 | max(-3, 1-3) = -2 | 1 | Extend subarray |
| 3 | 4 | max(4, -2+4) = 4 | 4 | Start new subarray |
| 4 | -1 | max(-1, 4-1) = 3 | 4 | Extend subarray |
| 5 | 2 | max(2, 3+2) = 5 | 5 | Extend subarray |
| 6 | 1 | max(1, 5+1) = 6 | 6 | Extend subarray |
| 7 | -5 | max(-5, 6-5) = 1 | 6 | Extend subarray |
| 8 | 4 | max(4, 1+4) = 5 | 6 | Extend subarray |

**Result:** `6` (subarray: [4,-1,2,1])

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func maxSubArray(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    // Initialize with first element
    maxSum := nums[0]
    currentSum := nums[0]
    
    // Process remaining elements
    for i := 1; i < len(nums); i++ {
        // Either start new subarray or extend existing one
        currentSum = max(nums[i], currentSum + nums[i])
        
        // Update maximum sum seen so far
        maxSum = max(maxSum, currentSum)
    }
    
    return maxSum
}

// Helper function to find maximum of two integers
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

### Alternative Solutions

#### **Brute Force Implementation**
```go
func maxSubArrayBruteForce(nums []int) int {
    maxSum := nums[0]
    
    // Check all possible subarrays
    for i := 0; i < len(nums); i++ {
        currentSum := 0
        for j := i; j < len(nums); j++ {
            currentSum += nums[j]
            maxSum = max(maxSum, currentSum)
        }
    }
    
    return maxSum
}
```

#### **Divide and Conquer**
```go
func maxSubArrayDivideConquer(nums []int) int {
    return maxSubArrayHelper(nums, 0, len(nums)-1)
}

func maxSubArrayHelper(nums []int, left, right int) int {
    if left == right {
        return nums[left]
    }
    
    mid := (left + right) / 2
    
    // Maximum subarray in left half
    leftMax := maxSubArrayHelper(nums, left, mid)
    
    // Maximum subarray in right half
    rightMax := maxSubArrayHelper(nums, mid+1, right)
    
    // Maximum subarray crossing the middle
    crossMax := maxCrossingSum(nums, left, mid, right)
    
    return max(max(leftMax, rightMax), crossMax)
}

func maxCrossingSum(nums []int, left, mid, right int) int {
    // Find maximum sum from mid to left
    leftSum := nums[mid]
    sum := nums[mid]
    for i := mid - 1; i >= left; i-- {
        sum += nums[i]
        leftSum = max(leftSum, sum)
    }
    
    // Find maximum sum from mid+1 to right
    rightSum := nums[mid+1]
    sum = nums[mid+1]
    for i := mid + 2; i <= right; i++ {
        sum += nums[i]
        rightSum = max(rightSum, sum)
    }
    
    return leftSum + rightSum
}
```

#### **Returning the Actual Subarray**
```go
func maxSubArrayWithIndices(nums []int) (int, int, int) {
    if len(nums) == 0 {
        return 0, 0, 0
    }
    
    maxSum := nums[0]
    currentSum := nums[0]
    start, end := 0, 0
    tempStart := 0
    
    for i := 1; i < len(nums); i++ {
        if currentSum < 0 {
            currentSum = nums[i]
            tempStart = i
        } else {
            currentSum += nums[i]
        }
        
        if currentSum > maxSum {
            maxSum = currentSum
            start = tempStart
            end = i
        }
    }
    
    return maxSum, start, end
}
```

### Notes / Variations

#### **Related Problems**
- **Maximum Product Subarray**: Find contiguous subarray with maximum product
- **Maximum Sum Circular Subarray**: Handle circular arrays
- **Maximum Sum of 3 Non-Overlapping Subarrays**: Find three non-overlapping subarrays
- **Maximum Subarray Sum After One Operation**: Modify one element to maximize sum

#### **ICPC Insights**
- **Edge Cases**: Handle arrays with all negative numbers
- **Overflow**: Be careful with integer overflow in large inputs
- **Memory**: Kadane's algorithm is optimal for space complexity
- **Implementation**: Practice the clean, one-pass implementation

#### **Go-Specific Optimizations**
```go
// Use math.Max for cleaner code (Go 1.21+)
import "math"

func maxSubArray(nums []int) int {
    maxSum := nums[0]
    currentSum := nums[0]
    
    for i := 1; i < len(nums); i++ {
        currentSum = int(math.Max(float64(nums[i]), float64(currentSum + nums[i])))
        maxSum = int(math.Max(float64(maxSum), float64(currentSum)))
    }
    
    return maxSum
}
```

#### **Real-World Applications**
- **Stock Trading**: Maximum profit from buying and selling stocks
- **Signal Processing**: Find the segment with maximum signal strength
- **Data Analysis**: Identify the most profitable time period
- **Game Development**: Find the best sequence of moves

### Testing

```go
func TestMaxSubArray(t *testing.T) {
    tests := []struct {
        nums     []int
        expected int
    }{
        {[]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}, 6},
        {[]int{1}, 1},
        {[]int{5, 4, -1, 7, 8}, 23},
        {[]int{-1}, -1},
        {[]int{-2, -1}, -1},
    }
    
    for _, test := range tests {
        result := maxSubArray(test.nums)
        if result != test.expected {
            t.Errorf("maxSubArray(%v) = %d, expected %d", 
                test.nums, result, test.expected)
        }
    }
}
```

### Visualization

```
Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        ↑
        Start new subarray (sum = -2)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
           ↑
           Start new subarray (sum = 1)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
              ↑
              Extend subarray (sum = -2)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                 ↑
                 Start new subarray (sum = 4)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                    ↑
                    Extend subarray (sum = 3)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                       ↑
                       Extend subarray (sum = 5)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                          ↑
                          Extend subarray (sum = 6) ← Maximum!

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                             ↑
                             Extend subarray (sum = 1)

Array: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
                                ↑
                                Extend subarray (sum = 5)
```
