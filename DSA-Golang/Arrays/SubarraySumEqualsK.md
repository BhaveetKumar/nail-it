# Subarray Sum Equals K

### Problem
Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

**Example:**
```
Input: nums = [1,1,1], k = 2
Output: 2

Input: nums = [1,2,3], k = 3
Output: 2
```

**Constraints:**
- 1 ≤ nums.length ≤ 2 × 10⁴
- -1000 ≤ nums[i] ≤ 1000
- -10⁷ ≤ k ≤ 10⁷

### Explanation

#### **Prefix Sum + Hash Map**
- Use prefix sum to calculate cumulative sum
- Use hash map to store frequency of prefix sums
- For each position, check if (current_sum - k) exists in map
- Time Complexity: O(n)
- Space Complexity: O(n)

### Golang Solution

```go
func subarraySum(nums []int, k int) int {
    count := 0
    sum := 0
    prefixSum := make(map[int]int)
    prefixSum[0] = 1 // Empty subarray has sum 0
    
    for _, num := range nums {
        sum += num
        if freq, exists := prefixSum[sum-k]; exists {
            count += freq
        }
        prefixSum[sum]++
    }
    
    return count
}
```

### Notes / Variations

#### **Related Problems**
- **Two Sum**: Find two numbers that sum to target
- **3Sum**: Find three numbers that sum to zero
- **Subarray Sum Divisible by K**: Find subarrays divisible by K
- **Continuous Subarray Sum**: Find subarray with sum multiple of k
