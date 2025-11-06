---
# Auto-generated front matter
Title: Twosum
LastUpdated: 2025-11-06T20:45:58.721476
Tags: []
Status: draft
---

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

**Detailed Explanation:**
The brute force approach is the most straightforward solution that checks every possible pair of numbers in the array. While simple to understand and implement, it's not efficient for large inputs.

**Algorithm Steps:**

1. **Nested Loop**: Use two nested loops to check all possible pairs
2. **Pair Generation**: For each element at index `i`, check all elements at index `j > i`
3. **Sum Check**: If `nums[i] + nums[j] == target`, return the indices
4. **No Solution**: If no pair is found, return null

**Why It Works:**

- **Complete Search**: Guarantees finding the solution if it exists
- **Simple Logic**: Easy to understand and implement
- **No Extra Space**: Uses only O(1) additional space

**Limitations:**

- **Time Complexity**: O(n²) makes it inefficient for large inputs
- **Redundant Checks**: Checks the same pairs multiple times
- **Scalability**: Performance degrades quadratically with input size

**Go Implementation:**

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

#### **Optimized Approach (Hash Map)**

**Detailed Explanation:**
The hash map approach is the optimal solution that trades space for time. It uses a hash map to store previously seen numbers and their indices, allowing us to find the complement in O(1) time.

**Algorithm Steps:**

1. **Initialize Map**: Create an empty hash map to store number-index pairs
2. **Single Pass**: Iterate through the array once
3. **Complement Check**: For each number, calculate `target - current_number`
4. **Lookup**: Check if the complement exists in the map
5. **Return or Store**: If found, return indices; otherwise, store current number

**Why It Works:**

- **Hash Map Lookup**: O(1) average case lookup time
- **Single Pass**: Processes each element exactly once
- **Space-Time Trade-off**: Uses O(n) space for O(n) time complexity

**Key Insights:**

- **Complement Strategy**: Instead of looking for pairs, look for complements
- **One-Pass Solution**: Can find the answer in a single iteration
- **Index Preservation**: Stores indices for the final result

**Go Implementation:**

```go
func twoSum(nums []int, target int) []int {
    seen := make(map[int]int)

    for i, num := range nums {
        complement := target - num

        if j, exists := seen[complement]; exists {
            return []int{j, i}
        }

        seen[num] = i
    }

    return nil
}
```

**Discussion Questions & Answers:**

**Q1: Why is the hash map approach more efficient than brute force?**

**Answer:** The hash map approach is more efficient because:

- **Time Complexity**: O(n) vs O(n²) - linear vs quadratic
- **Lookup Speed**: O(1) hash map lookup vs O(n) linear search
- **Single Pass**: Processes each element once vs multiple times
- **Scalability**: Performance scales linearly with input size
- **Real-world Performance**: Significantly faster for large inputs

**Q2: What are the trade-offs between the brute force and hash map approaches?**

**Answer:** Trade-offs include:

- **Space vs Time**: Brute force uses O(1) space but O(n²) time
- **Hash map uses O(n) space but O(n) time**
- **Implementation Complexity**: Brute force is simpler to implement
- **Memory Usage**: Hash map requires additional memory for the map
- **Hash Collisions**: Hash map performance can degrade with collisions

**Q3: How do you handle edge cases in the Two Sum problem?**

**Answer:** Edge cases to consider:

- **Empty Array**: Return null or empty result
- **Single Element**: No valid pair possible
- **No Solution**: Return null when no pair sums to target
- **Duplicate Numbers**: Handle cases where the same number appears twice
- **Negative Numbers**: Ensure the algorithm works with negative values
- **Large Numbers**: Handle integer overflow in sum calculations

### Dry Run

**Input:** `nums = [2,7,11,15]`, `target = 9`

| Step | i   | nums[i] | need = target - nums[i] | seen map | Action                  |
| ---- | --- | ------- | ----------------------- | -------- | ----------------------- |
| 1    | 0   | 2       | 9 - 2 = 7               | {}       | 7 not found, add {2: 0} |
| 2    | 1   | 7       | 9 - 7 = 2               | {2: 0}   | 2 found! Return [0, 1]  |

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
