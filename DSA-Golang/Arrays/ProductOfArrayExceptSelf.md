# Product of Array Except Self

### Problem

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operator.

**Example:**

```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

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
    rightProduct := 1
    for i := n - 1; i >= 0; i-- {
        result[i] = result[i] * rightProduct
        rightProduct *= nums[i]
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

#### **Handle Zeros**

```go
func productExceptSelfWithZeros(nums []int) []int {
    n := len(nums)
    result := make([]int, n)

    // Count zeros and calculate total product
    zeroCount := 0
    totalProduct := 1

    for _, num := range nums {
        if num == 0 {
            zeroCount++
        } else {
            totalProduct *= num
        }
    }

    for i, num := range nums {
        if zeroCount > 1 {
            result[i] = 0
        } else if zeroCount == 1 {
            if num == 0 {
                result[i] = totalProduct
            } else {
                result[i] = 0
            }
        } else {
            result[i] = totalProduct / num
        }
    }

    return result
}
```

### Complexity

- **Time Complexity:** O(n)
- **Space Complexity:** O(1) for optimal, O(n) for extra space
