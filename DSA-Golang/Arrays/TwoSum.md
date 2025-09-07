# Two Sum

### Problem
Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to the target.

**Example:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

**Constraints:**
- 2 ≤ nums.length ≤ 10⁴
- -10⁹ ≤ nums[i] ≤ 10⁹
- -10⁹ ≤ target ≤ 10⁹
- Only one valid answer exists

### Explanation

#### **Brute Force Approach**
- Check all possible pairs of numbers
- For each number at index `i`, check all numbers at index `j > i`
- Time Complexity: O(n²)
- Space Complexity: O(1)

#### **Optimized Approach (Hash Map)**
- Use a hash map to store visited numbers and their indices
- For each number, check if `target - current_number` exists in the map
- If found, return the indices; otherwise, add current number to map
- Time Complexity: O(n)
- Space Complexity: O(n)

### Dry Run

**Input:** `nums = [2,7,11,15]`, `target = 9`

| Step | i | nums[i] | need = target - nums[i] | seen map | Action |
|------|---|---------|------------------------|----------|---------|
| 1 | 0 | 2 | 9 - 2 = 7 | {} | 7 not found, add {2: 0} |
| 2 | 1 | 7 | 9 - 7 = 2 | {2: 0} | 2 found! Return [0, 1] |

**Result:** `[0, 1]`

### Complexity
- **Time Complexity:** O(n) - Single pass through the array
- **Space Complexity:** O(n) - Hash map stores at most n elements

### Golang Solution

```go
func twoSum(nums []int, target int) []int {
    // Create a map to store number -> index mapping
    seen := make(map[int]int)
    
    // Iterate through the array
    for i, num := range nums {
        // Calculate the complement
        complement := target - num
        
        // Check if complement exists in map
        if j, exists := seen[complement]; exists {
            return []int{j, i}
        }
        
        // Store current number and its index
        seen[num] = i
    }
    
    // No solution found (should not happen according to problem constraints)
    return nil
}
```

### Alternative Solutions

#### **Brute Force Implementation**
```go
func twoSumBruteForce(nums []int, target int) []int {
    for i := 0; i < len(nums); i++ {
        for j := i + 1; j < len(nums); j++ {
            if nums[i] + nums[j] == target {
                return []int{i, j}
            }
        }
    }
    return nil
}
```

#### **Two Pointers (for sorted array)**
```go
func twoSumTwoPointers(nums []int, target int) []int {
    // Note: This requires the array to be sorted
    // and returns values, not indices
    left, right := 0, len(nums)-1
    
    for left < right {
        sum := nums[left] + nums[right]
        if sum == target {
            return []int{nums[left], nums[right]}
        } else if sum < target {
            left++
        } else {
            right--
        }
    }
    return nil
}
```

### Notes / Variations

#### **Related Problems**
- **3Sum**: Find all unique triplets that sum to zero
- **4Sum**: Find all unique quadruplets that sum to target
- **Two Sum II**: Sorted array version
- **Two Sum III**: Design a data structure

#### **ICPC Insights**
- **Hash Map Optimization**: Use `make(map[int]int, n)` to pre-allocate capacity
- **Memory Management**: In contests, be mindful of map overhead
- **Edge Cases**: Handle duplicate numbers and negative targets
- **Input Size**: For very large inputs, consider space-time tradeoffs

#### **Go-Specific Optimizations**
```go
// Pre-allocate map capacity for better performance
seen := make(map[int]int, len(nums))

// Use struct{} for set-like operations (saves memory)
seen := make(map[int]struct{})
seen[value] = struct{}{}
```

#### **Real-World Applications**
- **Financial Systems**: Find pairs of transactions that sum to a target amount
- **Game Development**: Find combinations of items with specific values
- **Data Analysis**: Identify complementary data points
- **Cryptography**: Find key pairs in encryption algorithms

### Testing

```go
func TestTwoSum(t *testing.T) {
    tests := []struct {
        nums     []int
        target   int
        expected []int
    }{
        {[]int{2, 7, 11, 15}, 9, []int{0, 1}},
        {[]int{3, 2, 4}, 6, []int{1, 2}},
        {[]int{3, 3}, 6, []int{0, 1}},
    }
    
    for _, test := range tests {
        result := twoSum(test.nums, test.target)
        if !reflect.DeepEqual(result, test.expected) {
            t.Errorf("twoSum(%v, %d) = %v, expected %v", 
                test.nums, test.target, result, test.expected)
        }
    }
}
```
