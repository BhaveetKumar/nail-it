# Longest Increasing Subsequence (LIS)

### Problem
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements.

**Example:**
```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,18], therefore the length is 4.
```

**Constraints:**
- 1 ≤ nums.length ≤ 2500
- -10⁴ ≤ nums[i] ≤ 10⁴

### Explanation

#### **Brute Force Approach**
- Generate all possible subsequences
- Check which ones are increasing
- Return the length of the longest one
- Time Complexity: O(2ⁿ)
- Space Complexity: O(n)

#### **Dynamic Programming Approach (O(n²))**
- `dp[i]` represents the length of LIS ending at index `i`
- For each element, check all previous elements
- If previous element is smaller, extend the subsequence
- Time Complexity: O(n²)
- Space Complexity: O(n)

#### **Binary Search Approach (O(n log n))**
- Maintain an array `tails` where `tails[i]` is the smallest tail of all increasing subsequences of length `i+1`
- For each element, use binary search to find the position to replace
- Time Complexity: O(n log n)
- Space Complexity: O(n)

### Dry Run

**Input:** `nums = [10,9,2,5,3,7,101,18]`

#### **DP Approach (O(n²))**

| i | nums[i] | dp[i] | Explanation |
|---|---------|-------|-------------|
| 0 | 10 | 1 | Base case |
| 1 | 9 | 1 | No smaller previous element |
| 2 | 2 | 1 | No smaller previous element |
| 3 | 5 | 2 | Can extend from 2 (dp[2] + 1) |
| 4 | 3 | 2 | Can extend from 2 (dp[2] + 1) |
| 5 | 7 | 3 | Can extend from 5 or 3 (max(dp[3], dp[4]) + 1) |
| 6 | 101 | 4 | Can extend from 7 (dp[5] + 1) |
| 7 | 18 | 4 | Can extend from 7 (dp[5] + 1) |

**Result:** `4`

#### **Binary Search Approach (O(n log n))**

| Step | nums[i] | tails array | Action |
|------|---------|-------------|---------|
| 1 | 10 | [10] | Add 10 |
| 2 | 9 | [9] | Replace 10 with 9 |
| 3 | 2 | [2] | Replace 9 with 2 |
| 4 | 5 | [2, 5] | Add 5 |
| 5 | 3 | [2, 3] | Replace 5 with 3 |
| 6 | 7 | [2, 3, 7] | Add 7 |
| 7 | 101 | [2, 3, 7, 101] | Add 101 |
| 8 | 18 | [2, 3, 7, 18] | Replace 101 with 18 |

**Result:** `4` (length of tails array)

### Complexity
- **Time Complexity:** O(n log n) - Binary search approach
- **Space Complexity:** O(n) - Additional array for tails

### Golang Solution

#### **O(n²) DP Solution**
```go
func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    // dp[i] represents the length of LIS ending at index i
    dp := make([]int, len(nums))
    
    // Initialize all values to 1 (each element is a subsequence of length 1)
    for i := range dp {
        dp[i] = 1
    }
    
    maxLength := 1
    
    // For each element, check all previous elements
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            // If previous element is smaller, we can extend the subsequence
            if nums[j] < nums[i] {
                dp[i] = max(dp[i], dp[j]+1)
            }
        }
        maxLength = max(maxLength, dp[i])
    }
    
    return maxLength
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

#### **O(n log n) Binary Search Solution**
```go
func lengthOfLIS(nums []int) int {
    if len(nums) == 0 {
        return 0
    }
    
    // tails[i] is the smallest tail of all increasing subsequences of length i+1
    tails := make([]int, 0)
    
    for _, num := range nums {
        // Binary search for the position to insert/replace
        pos := binarySearch(tails, num)
        
        if pos == len(tails) {
            // num is larger than all elements in tails, extend the array
            tails = append(tails, num)
        } else {
            // Replace the element at position pos
            tails[pos] = num
        }
    }
    
    return len(tails)
}

func binarySearch(tails []int, target int) int {
    left, right := 0, len(tails)-1
    
    for left <= right {
        mid := (left + right) / 2
        if tails[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return left
}
```

### Alternative Solutions

#### **Recursive with Memoization**
```go
func lengthOfLIS(nums []int) int {
    memo := make(map[int]int)
    return lisHelper(nums, -1, 0, memo)
}

func lisHelper(nums []int, prevIndex, currIndex int, memo map[int]int) int {
    if currIndex == len(nums) {
        return 0
    }
    
    key := prevIndex*10000 + currIndex
    if val, exists := memo[key]; exists {
        return val
    }
    
    // Option 1: Don't take current element
    taken := lisHelper(nums, prevIndex, currIndex+1, memo)
    
    // Option 2: Take current element (if it's larger than previous)
    if prevIndex == -1 || nums[currIndex] > nums[prevIndex] {
        taken = max(taken, 1+lisHelper(nums, currIndex, currIndex+1, memo))
    }
    
    memo[key] = taken
    return taken
}
```

#### **Returning the Actual LIS**
```go
func lengthOfLISWithSequence(nums []int) (int, []int) {
    if len(nums) == 0 {
        return 0, []int{}
    }
    
    dp := make([]int, len(nums))
    parent := make([]int, len(nums))
    
    for i := range dp {
        dp[i] = 1
        parent[i] = -1
    }
    
    maxLength := 1
    maxIndex := 0
    
    for i := 1; i < len(nums); i++ {
        for j := 0; j < i; j++ {
            if nums[j] < nums[i] && dp[j]+1 > dp[i] {
                dp[i] = dp[j] + 1
                parent[i] = j
            }
        }
        if dp[i] > maxLength {
            maxLength = dp[i]
            maxIndex = i
        }
    }
    
    // Reconstruct the sequence
    sequence := make([]int, 0, maxLength)
    for maxIndex != -1 {
        sequence = append(sequence, nums[maxIndex])
        maxIndex = parent[maxIndex]
    }
    
    // Reverse the sequence
    for i, j := 0, len(sequence)-1; i < j; i, j = i+1, j-1 {
        sequence[i], sequence[j] = sequence[j], sequence[i]
    }
    
    return maxLength, sequence
}
```

### Notes / Variations

#### **Related Problems**
- **Longest Decreasing Subsequence**: Reverse the array and find LIS
- **Longest Bitonic Subsequence**: LIS + LDS
- **Russian Doll Envelopes**: 2D LIS problem
- **Maximum Length of Pair Chain**: Interval scheduling
- **Number of Longest Increasing Subsequence**: Count LIS

#### **ICPC Insights**
- **Binary Search Optimization**: Always prefer O(n log n) over O(n²)
- **Memory Management**: Use space-efficient implementations
- **Edge Cases**: Handle empty arrays and single elements
- **Implementation Speed**: Practice the binary search approach

#### **Go-Specific Optimizations**
```go
// Use sort.Search for binary search
import "sort"

func lengthOfLIS(nums []int) int {
    tails := make([]int, 0)
    
    for _, num := range nums {
        pos := sort.Search(len(tails), func(i int) bool {
            return tails[i] >= num
        })
        
        if pos == len(tails) {
            tails = append(tails, num)
        } else {
            tails[pos] = num
        }
    }
    
    return len(tails)
}
```

#### **Real-World Applications**
- **Stock Trading**: Find longest increasing price sequence
- **DNA Sequencing**: Find longest common subsequence
- **Project Scheduling**: Find longest chain of dependent tasks
- **Data Analysis**: Find longest trend in time series data

### Testing

```go
func TestLengthOfLIS(t *testing.T) {
    tests := []struct {
        nums     []int
        expected int
    }{
        {[]int{10, 9, 2, 5, 3, 7, 101, 18}, 4},
        {[]int{0, 1, 0, 3, 2, 3}, 4},
        {[]int{7, 7, 7, 7, 7, 7, 7}, 1},
        {[]int{1}, 1},
        {[]int{1, 3, 6, 7, 9, 4, 10, 5, 6}, 6},
    }
    
    for _, test := range tests {
        result := lengthOfLIS(test.nums)
        if result != test.expected {
            t.Errorf("lengthOfLIS(%v) = %d, expected %d", 
                test.nums, result, test.expected)
        }
    }
}
```

### Visualization

```
Input: [10, 9, 2, 5, 3, 7, 101, 18]

DP Approach:
Index: 0  1  2  3  4  5  6   7
Value: 10 9  2  5  3  7  101 18
DP:    1  1  1  2  2  3  4   4

Binary Search Approach:
Step 1: [10]
Step 2: [9]     (replace 10)
Step 3: [2]     (replace 9)
Step 4: [2, 5]  (add 5)
Step 5: [2, 3]  (replace 5)
Step 6: [2, 3, 7] (add 7)
Step 7: [2, 3, 7, 101] (add 101)
Step 8: [2, 3, 7, 18] (replace 101)

Final LIS: [2, 3, 7, 18] with length 4
```

### Mathematical Insight

**Why does binary search work?**

The key insight is that we only need to maintain the **smallest tail** of each possible length. This is because:

1. If we have two increasing subsequences of the same length, we only care about the one with the smaller tail
2. A smaller tail gives us more opportunities to extend the subsequence
3. The `tails` array is always sorted, making binary search possible

**Example:**
- Length 1: [2] is better than [10] because 2 < 10
- Length 2: [2, 3] is better than [2, 5] because 3 < 5
- Length 3: [2, 3, 7] is better than [2, 3, 101] because 7 < 101
