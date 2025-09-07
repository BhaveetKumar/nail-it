# Product of Array Except Self

### Problem
Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operator.

**Example:**
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Explanation: 
answer[0] = 2*3*4 = 24
answer[1] = 1*3*4 = 12
answer[2] = 1*2*4 = 8
answer[3] = 1*2*3 = 6

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

**Constraints:**
- 2 ≤ nums.length ≤ 10⁵
- -30 ≤ nums[i] ≤ 30
- The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer

### Explanation

#### **Two Pass Approach**
- First pass: Calculate left products (product of all elements to the left)
- Second pass: Calculate right products and multiply with left products
- Time Complexity: O(n)
- Space Complexity: O(1) excluding output array

### Dry Run

**Input:** `nums = [1,2,3,4]`

#### **First Pass (Left Products)**
| i | left[i] | Calculation |
|---|---------|-------------|
| 0 | 1 | 1 (no left elements) |
| 1 | 1 | 1 * 1 = 1 |
| 2 | 2 | 1 * 2 = 2 |
| 3 | 6 | 2 * 3 = 6 |

#### **Second Pass (Right Products)**
| i | right | left[i] | result[i] |
|---|-------|---------|-----------|
| 3 | 1 | 6 | 6 * 1 = 6 |
| 2 | 4 | 2 | 2 * 4 = 8 |
| 1 | 12 | 1 | 1 * 12 = 12 |
| 0 | 24 | 1 | 1 * 24 = 24 |

**Result:** `[24,12,8,6]`

### Complexity
- **Time Complexity:** O(n) - Two passes through the array
- **Space Complexity:** O(1) - Only using constant extra space

### Golang Solution

```go
func productExceptSelf(nums []int) []int {
    n := len(nums)
    result := make([]int, n)
    
    // First pass: calculate left products
    result[0] = 1
    for i := 1; i < n; i++ {
        result[i] = result[i-1] * nums[i-1]
    }
    
    // Second pass: calculate right products and multiply
    right := 1
    for i := n - 1; i >= 0; i-- {
        result[i] = result[i] * right
        right *= nums[i]
    }
    
    return result
}
```

### Alternative Solutions

#### **Using Extra Space**
```go
func productExceptSelfExtraSpace(nums []int) []int {
    n := len(nums)
    left := make([]int, n)
    right := make([]int, n)
    result := make([]int, n)
    
    // Calculate left products
    left[0] = 1
    for i := 1; i < n; i++ {
        left[i] = left[i-1] * nums[i-1]
    }
    
    // Calculate right products
    right[n-1] = 1
    for i := n - 2; i >= 0; i-- {
        right[i] = right[i+1] * nums[i+1]
    }
    
    // Calculate result
    for i := 0; i < n; i++ {
        result[i] = left[i] * right[i]
    }
    
    return result
}
```

### Notes / Variations

#### **Related Problems**
- **Trapping Rain Water**: Similar two-pass approach
- **Maximum Product Subarray**: Find maximum product
- **Product of Array Except Self II**: Handle zeros differently
- **Array Product**: Calculate product with constraints

#### **ICPC Insights**
- **Two Pass**: Use two passes to avoid division
- **Space Optimization**: Use output array for intermediate results
- **Zero Handling**: Handle zeros in the array
- **Overflow**: Be careful with integer overflow
